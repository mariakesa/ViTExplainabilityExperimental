#!/usr/bin/env python3
"""Compare ViT stationary vs incoming-attention encoding of mouse population responses.

Uses matched scene_000 ... scene_117 order. All PCA, feature scaling, and alpha
selection are confined to training data. Uses image-level CV, never trial-level.

Install: pip install numpy scipy scikit-learn matplotlib
Run: python encode_mouse_from_markov.py
"""
import argparse
import csv
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/maria/ViTExplainabilityExperimental/data')
ATT = ROOT / 'vit_attention/vit_attention_float16.npy'
NEURAL = Path('/home/maria/SelfStudyThesis/data/allen_natural_scenes_processed_data.npy')
ALPHAS = np.logspace(-2, 5, 9)


def extract_features(path, destination, damping=0.995, tol=1e-7, max_iter=200):
    """Returns arrays (images,layers,heads,tokens). Cached, for re-use."""
    if destination.exists():
        with np.load(destination) as saved:
            if np.isclose(float(saved['damping']), damping):
                print(f'Loading cached features: {destination}', flush=True)
                return saved['stationary'], saved['incoming']
    att = np.load(path, mmap_mode='r')
    if att.ndim != 5 or att.shape[-1] != att.shape[-2]:
        raise ValueError(f'Unexpected attention shape: {att.shape}')
    n, n_layers, n_heads, tokens, _ = att.shape
    stationary = np.empty((n,n_layers,n_heads,tokens), dtype=np.float32)
    incoming = np.empty_like(stationary)
    # Regularized chain T=(1-damping)*uniform + damping*attention.
    # Positivity/uniqueness survives float16 zeros and convergence is stabilized.
    teleport = (1.0-damping)/tokens
    for layer in range(n_layers):
        for image in range(n):
            p = np.asarray(att[image,layer], dtype=np.float32).copy()  # heads,tokens,tokens
            p /= np.maximum(p.sum(axis=-1,keepdims=True),1e-20)
            inc = p.mean(axis=1)  # average incoming attention for uniform queries
            incoming[image,layer] = inc / inc.sum(axis=-1,keepdims=True)
            pi = np.full((n_heads,tokens),1.0/tokens,dtype=np.float32)
            for k in range(max_iter):
                nxt = damping*np.einsum('ht,htu->hu',pi,p,optimize=True)+teleport
                if np.max(np.abs(nxt-pi).sum(axis=1)) < tol:
                    pi=nxt
                    break
                pi=nxt
            else:
                print(f'WARNING: convergence cap at image={image}, layer={layer}; check residual',flush=True)
            stationary[image,layer]=pi/ pi.sum(axis=1,keepdims=True)
        print(f'Extracted layer {layer+1}/{n_layers}',flush=True)
    destination.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(destination,stationary=stationary,incoming=incoming,damping=np.array(damping))
    return stationary,incoming


def load_neural(path,n):
    obj = np.load(path, allow_pickle=True).item()
    x = np.asarray(obj['X'],dtype=np.float32)
    if x.shape[0] == n and x.shape[1] != n:
        pass
    elif x.shape[1] == n and x.shape[0] != n:
        x=x.T
    else:
        raise ValueError(f'Cannot identify image axis of neural X: {x.shape}; expected one axis {n}')
    if not np.isfinite(x).all():
        raise ValueError('Neural responses contain nonfinite entries')
    return x


def train_pca(y_train, k):
    # PCA is fit ONLY on the current training split.
    n_components=min(k,y_train.shape[0]-1,y_train.shape[1])
    pca=PCA(n_components=n_components,svd_solver='randomized',random_state=42,n_oversamples=5)
    train_scores=pca.fit_transform(y_train)
    return pca,train_scores


def kernel_matrices(z_train,z_test):
    sc=StandardScaler().fit(z_train)
    tr=sc.transform(z_train).astype(np.float64)
    te=sc.transform(z_test).astype(np.float64)
    # Division makes alphas comparable when varying feature dimensions.
    divisor=tr.shape[1]
    return (tr@tr.T)/divisor, (te@tr.T)/divisor


def solve_kernel(Ktr, Kte, targets,alpha):
    return Kte @ np.linalg.solve(Ktr + alpha*np.eye(Ktr.shape[0]),targets)


def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--attention',type=Path,default=ATT)
    ap.add_argument('--neural',type=Path,default=NEURAL)
    ap.add_argument('--output',type=Path,default=ROOT/'vit_attention/encoding_experiment')
    ap.add_argument('--outer-folds',type=int,default=5,help='5 by default; 118 for LOO, slower')
    ap.add_argument('--inner-folds',type=int,default=3)
    ap.add_argument('--n-pcs',type=int,default=10)
    ap.add_argument('--damping',type=float,default=.995)
    ap.add_argument('--region-indices',type=Path,default=None,help='Optional JSON mapping region names to 0-based neuron column indices')
    args=ap.parse_args()
    args.output.mkdir(parents=True,exist_ok=True)
    zstat, zinc=extract_features(args.attention,args.output/'markov_features.npz',damping=args.damping)
    n,nlayers,nheads,tokens=zstat.shape
    y_all=load_neural(args.neural,n)
    regions={'ALL':np.arange(y_all.shape[1])}
    if args.region_indices is not None:
        mappings=json.loads(args.region_indices.read_text())
        regions={str(name):np.asarray(idx,dtype=int) for name,idx in mappings.items()}
    if not 2 <= args.outer_folds <= n:
        raise ValueError('outer folds must be between 2 and number of images')
    if args.outer_folds == n:
        from sklearn.model_selection import LeaveOneOut
        outer=LeaveOneOut()
    else:
        outer=KFold(n_splits=args.outer_folds,shuffle=True,random_state=2026)
    all_results=[]
    for region,columns in regions.items():
        y=y_all[:,columns].astype(np.float64)
        print(f'\nREGION {region} Y={y.shape}',flush=True)
        split=list(outer.split(np.arange(n)))
        # Target PCA's per outer and inner split are independent of image features,
        # so compute once, then reuse across all layers and representations.
        foldinfo=[]
        for fold,(outer_tr,outer_te) in enumerate(split):
            pca,yscore=train_pca(y[outer_tr],args.n_pcs)
            ymean=y[outer_tr].mean(axis=0)
            # Independent inner PCA: no neural information from validation stimuli leaks in.
            inner=[]
            for itr,iva in KFold(n_splits=args.inner_folds,shuffle=True,random_state=17).split(outer_tr):
                train_ids,valid_ids=outer_tr[itr],outer_tr[iva]
                ipca,train_scores=train_pca(y[train_ids],args.n_pcs)
                inner.append((train_ids,valid_ids,ipca,train_scores))
            foldinfo.append((outer_tr,outer_te,pca,yscore,ymean,inner))
        for layer in range(nlayers):
            for name,zall in [('stationary',zstat),('incoming',zinc)]:
                z = zall[:,layer].reshape(n,nheads*tokens)
                predictions=np.empty_like(y)
                selected=[]
                for fold,(otr,ote,pca,yscore,ymean,inner) in enumerate(foldinfo):
                    scores=np.zeros(len(ALPHAS),dtype=np.float64)
                    for itr,iva,ipca,itarget in inner:
                        ktr,kva=kernel_matrices(z[itr],z[iva])
                        for ai,alpha in enumerate(ALPHAS):
                            ypred=ipca.inverse_transform(solve_kernel(ktr,kva,itarget,alpha))
                            scores[ai]+=np.sum((y[iva]-ypred)**2)
                    alpha=float(ALPHAS[np.argmin(scores)])
                    selected.append(alpha)
                    ktr,kte=kernel_matrices(z[otr],z[ote])
                    predictions[ote]=pca.inverse_transform(solve_kernel(ktr,kte,yscore,alpha))
                # Every image was predicted without using its own neural response.
                # Centered on a global mean for standardized reporting. Split-safe
                # mean-predictor baseline is separately accumulated below.
                sse=np.sum((y-predictions)**2)
                # Compute matched training-mean baseline for every held-out example.
                baseline=np.empty_like(y)
                for otr,ote,pca,yscore,ymean,inner in foldinfo:
                    baseline[ote]=ymean
                baseline_sse=np.sum((y-baseline)**2)
                r2=1.0-sse/max(baseline_sse,1e-20)
                result={'region':region,'layer':layer,'method':name,'r2_vs_train_mean':r2,
                        'model_sse':sse,'baseline_sse':baseline_sse,
                        'median_alpha':float(np.median(selected))}
                all_results.append(result)
                print(f' layer {layer:02d} {name:10s} R2={r2:+.5f} median_alpha={np.median(selected):.3g}',flush=True)
        subset=[r for r in all_results if r['region']==region]
        r_stat=np.array([next(r['r2_vs_train_mean'] for r in subset if r['layer']==i and r['method']=='stationary') for i in range(nlayers)])
        r_inc=np.array([next(r['r2_vs_train_mean'] for r in subset if r['layer']==i and r['method']=='incoming') for i in range(nlayers)])
        fig,ax=plt.subplots(figsize=(9,5))
        ax.plot(np.arange(nlayers)+1,r_stat,'o-',label='Stationary distribution')
        ax.plot(np.arange(nlayers)+1,r_inc,'s-',label='Mean incoming attention')
        ax.axhline(0,color='gray',linestyle='--',linewidth=1)
        ax.set(xlabel='ViT layer (1-based)',ylabel='Out-of-fold R² vs training-mean baseline',
               title=f'Neural encoding from ViT attention — {region}')
        ax.legend(); fig.tight_layout(); fig.savefig(args.output/f'encoding_{region}.png',dpi=170);plt.close(fig)
    with (args.output/'encoding_results.csv').open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(all_results[0]));writer.writeheader();writer.writerows(all_results)
    print(f'\nDone. Results: {args.output}/encoding_results.csv',flush=True)
    print('NOTE: 10-PC response reconstruction limits maximum possible performance.\n'
          'A negative R² means worse than predicting the training-set mean.',flush=True)

if __name__=='__main__':
    main()
