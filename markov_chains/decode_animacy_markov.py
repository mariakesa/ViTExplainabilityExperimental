#!/usr/bin/env python3
"""Decode animal vs non-animal from ViT attention stationary distributions.

Matches scene_000.png ... scene_117.png to neural-file stimulus rows (in that
order). Labels follow the user's existing convention: label == 0 -> animal.

Uses nested stratified CV. Each layer is evaluated independently; the
stationary vs incoming-attention comparison uses identical splits.

Install: pip install numpy scipy scikit-learn matplotlib
Run: python decode_animacy_markov.py
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path('/home/maria/ViTExplainabilityExperimental/data')
ATT = ROOT / 'vit_attention/vit_attention_float16.npy'
NEURAL = Path('/home/maria/SelfStudyThesis/data/allen_natural_scenes_processed_data.npy')
OUTPUT = ROOT / 'vit_attention/animacy_decoding'
DEFAULT_CACHE = ROOT / 'vit_attention/encoding_experiment/markov_features.npz'


def extract_markov(attention_path, cache_path, damping=0.995, tol=1e-7, max_iter=300):
    """Return stationary and incoming (images, layers, heads, tokens) float32.

    Uses teleport-regularized transition matrix T = damping*P + (1-damping)*U.
    Cached arrays are only reused if damping agrees.
    """
    if cache_path.exists():
        with np.load(cache_path) as cached:
            if (all(key in cached for key in ('stationary', 'incoming', 'damping'))
                    and np.isclose(float(cached['damping']), damping)):
                print(f'Using cached Markov features: {cache_path}', flush=True)
                return cached['stationary'], cached['incoming']
    print('Computing Markov features from attention tensor (once)...', flush=True)
    att = np.load(attention_path, mmap_mode='r')
    if att.ndim != 5 or att.shape[-1] != att.shape[-2]:
        raise ValueError(f'Expected (images,layers,heads,tokens,tokens), got {att.shape}')
    n, layers, heads, tokens, _ = att.shape
    stationary = np.empty((n, layers, heads, tokens), dtype=np.float32)
    incoming = np.empty_like(stationary)
    teleport = (1-damping)/tokens
    for layer in range(layers):
        for i in range(n):
            p = np.asarray(att[i, layer], dtype=np.float32).copy()
            if not np.isfinite(p).all():
                raise ValueError(f'Non-finite attention in image {i}, layer {layer}')
            p /= np.maximum(p.sum(axis=-1, keepdims=True), 1e-20)
            incoming[i, layer] = p.mean(axis=1)
            pi = np.full((heads, tokens), 1/tokens, dtype=np.float32)
            for step in range(max_iter):
                new = damping*np.einsum('ht,htu->hu', pi, p, optimize=True) + teleport
                residual = np.max(np.abs(new-pi).sum(axis=1))
                pi = new
                if residual < tol:
                    break
            else:
                print(f'WARNING: power method hit {max_iter} iterations: image={i}, layer={layer}, '
                      f'residual={residual:.2g}', flush=True)
            stationary[i, layer] = pi / pi.sum(axis=1, keepdims=True)
        print(f'  Markov features: layer {layer+1}/{layers}', flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(cache_path, stationary=stationary, incoming=incoming,
                        damping=np.array(damping, dtype=np.float64))
    return stationary, incoming


def load_labels(neural_path, n):
    data = np.load(neural_path, allow_pickle=True).item()
    raw = np.asarray(data['stimulus_metadata']['label']).reshape(-1)
    if len(raw) != n:
        raise ValueError(f'Label count {len(raw)} != number of images {n}')
    y = (raw == 0).astype(int)
    if len(np.unique(y)) != 2:
        raise ValueError('Labels are not binary after label==0 mapping')
    return y, raw


def make_model(n_pcs):
    steps = [('scale', StandardScaler())]
    if n_pcs > 0:
        steps.append(('pca', PCA(n_components=n_pcs, svd_solver='randomized', random_state=42)))
    steps.append(('logistic', LogisticRegression(penalty='l2', solver='liblinear', max_iter=3000)))
    return Pipeline(steps)


def evaluate(z, y, folds, inner_folds, pcs, C_values, seed):
    out = np.empty(len(y), dtype=np.float64)
    chosen = []
    for fold_id, (train, test) in enumerate(folds, start=1):
        n_pcs = min(pcs, len(train)-1) if pcs > 0 else 0
        pipeline = make_model(n_pcs)
        inner = StratifiedKFold(n_splits=inner_folds, shuffle=True, random_state=seed+17)
        grid = GridSearchCV(pipeline,
                            {'logistic__C': C_values},
                            scoring='balanced_accuracy', cv=inner,
                            n_jobs=1, refit=True)
        grid.fit(z[train], y[train])
        out[test] = grid.predict_proba(z[test])[:, 1]
        chosen.append(float(grid.best_params_['logistic__C']))
    pred = (out >= 0.5).astype(int)
    cm = confusion_matrix(y, pred, labels=[0, 1])
    return {
        'accuracy': float(accuracy_score(y, pred)),
        'balanced_accuracy': float(balanced_accuracy_score(y, pred)),
        'auc': float(roc_auc_score(y, out)),
        'tn': int(cm[0, 0]), 'fp': int(cm[0, 1]),
        'fn': int(cm[1, 0]), 'tp': int(cm[1, 1]),
        'median_C': float(np.median(chosen)),
        'scores': out,
        'predictions': pred,
        'chosen_C': chosen,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--attention', type=Path, default=ATT)
    parser.add_argument('--neural', type=Path, default=NEURAL)
    parser.add_argument('--cache', type=Path, default=DEFAULT_CACHE)
    parser.add_argument('--output', type=Path, default=OUTPUT)
    parser.add_argument('--outer-folds', type=int, default=5, help='5 (default); use 118 for LOO')
    parser.add_argument('--inner-folds', type=int, default=3)
    parser.add_argument('--pcs', type=int, default=20,
                        help='PCA features per layer, fitted inside CV (default 20; 0 disables)')
    parser.add_argument('--damping', type=float, default=0.995)
    parser.add_argument('--stationary-only', action='store_true',
                        help='Skip ordinary incoming-attention baseline')
    parser.add_argument('--seed', type=int, default=2026)
    args = parser.parse_args()
    if not (0 < args.damping < 1):
        parser.error('--damping must be between 0 and 1')
    args.output.mkdir(parents=True, exist_ok=True)

    zstat, zinc = extract_markov(args.attention, args.cache, damping=args.damping)
    n, n_layers, n_heads, tokens = zstat.shape
    y, raw = load_labels(args.neural, n)
    print(f'Features: {n} images, {n_layers} layers, {n_heads} heads, {tokens} tokens', flush=True)
    print(f'Animacy labels from neural data: animal={y.sum()}, non-animal={n-y.sum()}', flush=True)
    print('ASSUMPTION: neural-file label row i corresponds to scene_{i:03d}.png.', flush=True)
    if args.outer_folds == n:
        from sklearn.model_selection import LeaveOneOut
        outer = LeaveOneOut()
    elif 2 <= args.outer_folds <= min(np.bincount(y)):
        outer = StratifiedKFold(n_splits=args.outer_folds, shuffle=True, random_state=args.seed)
    else:
        parser.error('--outer-folds must be between 2 and the smallest class size, or exactly 118')
    folds = list(outer.split(np.arange(n), y))
    if args.inner_folds < 2:
        parser.error('--inner-folds must be at least 2')
    C_values = np.logspace(-3, 2, 6)
    feature_sets = [('stationary', zstat)]
    if not args.stationary_only:
        feature_sets.append(('incoming', zinc))

    rows = []
    out_of_fold = {'y_true': y}
    for layer in range(n_layers):
        for name, tensor in feature_sets:
            z = np.asarray(tensor[:, layer], dtype=np.float64).reshape(n, -1)
            result = evaluate(z, y, folds, args.inner_folds, args.pcs, C_values, args.seed)
            rows.append({k: result[k] for k in ('accuracy', 'balanced_accuracy', 'auc',
                                                'tn', 'fp', 'fn', 'tp', 'median_C')}
                        | {'layer': layer, 'method': name})
            out_of_fold[f'{name}_layer_{layer:02d}_prob'] = result['scores']
            out_of_fold[f'{name}_layer_{layer:02d}_pred'] = result['predictions']
            print(f'layer {layer:02d} | {name:10s} | accuracy={result["accuracy"]:.3f} '
                  f'balanced={result["balanced_accuracy"]:.3f} '
                  f'AUC={result["auc"]:.3f} median_C={result["median_C"]:g}', flush=True)

    with (args.output/'animacy_decoding_by_layer.csv').open('w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['layer', 'method', 'accuracy', 'balanced_accuracy',
                                               'auc', 'tn', 'fp', 'fn', 'tp', 'median_C'])
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(args.output/'oof_predictions.npz', **out_of_fold)

    fig, axes = plt.subplots(2, 1, figsize=(10, 9), sharex=True)
    for name, _ in feature_sets:
        data = [r for r in rows if r['method'] == name]
        axes[0].plot([r['layer'] + 1 for r in data], [r['balanced_accuracy'] for r in data],
                     marker='o', label=name)
        axes[1].plot([r['layer'] + 1 for r in data], [r['auc'] for r in data],
                     marker='o', label=name)
    axes[0].axhline(0.5, ls='--', alpha=0.6, label='chance (balanced accuracy)')
    axes[1].axhline(0.5, ls='--', alpha=0.6)
    axes[0].set_ylabel('Out-of-fold balanced accuracy')
    axes[1].set_ylabel('Out-of-fold ROC AUC')
    axes[1].set_xlabel('ViT layer (1–12)')
    axes[1].set_xticks(range(1, n_layers + 1))
    for ax in axes:
        ax.grid(alpha=0.2)
        ax.legend()
    fig.suptitle('Animacy decoding from attention-derived Markov features')
    fig.tight_layout()
    fig.savefig(args.output/'animacy_decoding_by_layer.png', dpi=180)
    plt.close(fig)
    metadata = {'attention_path': str(args.attention), 'neural_path': str(args.neural),
                'feature_cache': str(args.cache), 'n_images': n, 'n_layers': n_layers,
                'n_heads': n_heads, 'tokens': tokens, 'damping': args.damping,
                'outer_folds': args.outer_folds, 'inner_folds': args.inner_folds,
                'pcs': args.pcs, 'C_grid': C_values.tolist(), 'seed': args.seed,
                'positive_class': 'label == 0 (animal)',
                'assumed_alignment': 'scene_000 through scene_117 same order as neural metadata',
                'note': 'Layers are explored on the same dataset: selecting the best layer after '
                        'inspection makes its CV score optimistic for a selected-layer claim.'}
    (args.output/'run_metadata.json').write_text(json.dumps(metadata, indent=2))
    print('\nSaved:', args.output, flush=True)
    print('NOTE: Best-of-12 layer performance is exploratory; use an outer layer-selection '
          'procedure or independent test set for an unbiased best-layer claim.', flush=True)


if __name__ == '__main__':
    main()
