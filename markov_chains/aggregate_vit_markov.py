#!/usr/bin/env python3
"""Aggregate stationary Markov-chain statistics across all ViT images/layers/heads.

Default: analyzes all 118 x 12 x 12 attention matrices without loading the full
1.23 GiB array into RAM. Optional --mean-spectrum computes eigenvalues of the
*across-image averaged* transition matrix at each (layer, head), not an average
of individual spectra.

Run:
    python aggregate_vit_markov.py
    python aggregate_vit_markov.py --mean-spectrum
"""
import argparse
import csv
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

DEFAULT_INPUT = Path('/home/maria/ViTExplainabilityExperimental/data/vit_attention/vit_attention_float16.npy')


def stationary_batch(p, tolerance=2e-8, max_iter=400):
    """Compute left stationary distributions for N row-stochastic matrices.
    p shape (N,T,T), float32. Uses iteration, optionally with Cesaro-style fallback
    via scipy eig if convergence is slow (not ordinarily needed for softmax attention).
    """
    n, t, _ = p.shape
    x = np.full((n, t), 1.0/t, dtype=np.float32)
    converged = np.zeros(n, dtype=bool)
    for it in range(max_iter):
        y = np.einsum('ni,nij->nj', x, p, optimize=True)
        err = np.abs(y - x).sum(axis=1)
        x = y
        if np.all(err < tolerance):
            converged[:] = True
            break
    else:
        converged = err < tolerance
    x /= x.sum(axis=1, keepdims=True)
    residual = np.abs(np.einsum('ni,nij->nj', x, p, optimize=True) - x).sum(axis=1)
    return x, residual, it + 1


def save_heatmap(data, title, label, destination, fmt='.3f'):
    fig, ax = plt.subplots(figsize=(11, 6.2))
    im = ax.imshow(data, aspect='auto', interpolation='nearest', cmap='viridis')
    ax.set_xticks(range(data.shape[1]))
    ax.set_xticklabels(range(data.shape[1]))
    ax.set_yticks(range(data.shape[0]))
    ax.set_yticklabels(range(data.shape[0]))
    ax.set_xlabel('Attention head (0-based)')
    ax.set_ylabel('Transformer layer (0-based)')
    ax.set_title(title)
    cbar = fig.colorbar(im, ax=ax)
    cbar.set_label(label)
    fig.tight_layout()
    fig.savefig(destination, dpi=180)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--output', type=Path, default=None)
    ap.add_argument('--mean-spectrum', action='store_true', help='Compute nontrivial eigenvalue magnitudes for each across-image mean transition matrix (144 eig problems).')
    ap.add_argument('--max-iter', type=int, default=400)
    args = ap.parse_args()

    array = np.load(args.input, mmap_mode='r')
    if array.ndim != 5 or array.shape[-1] != array.shape[-2]:
        raise ValueError(f'Expected (images,layers,heads,tokens,tokens), got {array.shape}')
    n_images, n_layers, n_heads, n_tokens, _ = array.shape
    if n_tokens != 197:
        print(f'Warning: expected 197 tokens, got {n_tokens}; token 0 still treated as CLS')
    outdir = args.output or args.input.parent / 'markov_aggregate'
    outdir.mkdir(parents=True, exist_ok=True)

    shape = (n_images, n_layers, n_heads)
    cls_mass = np.empty(shape, dtype=np.float32)
    patch_entropy = np.empty(shape, dtype=np.float32)
    effective_patches = np.empty(shape, dtype=np.float32)
    max_patch_probability = np.empty(shape, dtype=np.float32)
    residuals = np.empty(shape, dtype=np.float32)
    mean_patch_maps = np.empty((n_layers, n_heads, n_tokens - 1), dtype=np.float32)
    spectral_slem = np.full((n_layers, n_heads), np.nan, dtype=np.float64)

    for layer in range(n_layers):
        for head in range(n_heads):
            p = np.asarray(array[:, layer, head], dtype=np.float32)
            if not np.isfinite(p).all() or np.any(p < 0):
                raise ValueError(f'Bad transition probabilities at layer={layer}, head={head}')
            # Half-precision storage may change row sums slightly.
            p /= p.sum(axis=-1, keepdims=True)
            pi, residual, iterations = stationary_batch(p, max_iter=args.max_iter)
            residuals[:, layer, head] = residual
            cls_mass[:, layer, head] = pi[:, 0]
            patch_mass = pi[:, 1:]
            patch_cond = patch_mass / np.maximum(patch_mass.sum(axis=1, keepdims=True), 1e-20)
            patch_entropy[:, layer, head] = -np.sum(patch_cond * np.log(np.clip(patch_cond, 1e-20, None)), axis=1) / np.log(n_tokens-1)
            effective_patches[:, layer, head] = 1.0 / np.sum(patch_cond ** 2, axis=1)
            max_patch_probability[:, layer, head] = patch_cond.max(axis=1)
            mean_patch_maps[layer, head] = patch_cond.mean(axis=0)
            if args.mean_spectrum:
                from scipy.linalg import eigvals
                mean_p = p.mean(axis=0, dtype=np.float64)
                mean_p /= mean_p.sum(axis=1, keepdims=True)
                eigenvalues = eigvals(mean_p)
                magnitudes = np.sort(np.abs(eigenvalues))[::-1]
                spectral_slem[layer, head] = magnitudes[1]
            print(f'layer {layer:02d} head {head:02d}: CLS={cls_mass[:,layer,head].mean():.4f} '
                  f'entropy={patch_entropy[:,layer,head].mean():.4f} '
                  f'effective_patches={effective_patches[:,layer,head].mean():.1f} '
                  f'max_residual={residual.max():.2e} iter={iterations}', flush=True)

    aggregated = {
        'mean_cls_mass': cls_mass.mean(axis=0),
        'mean_normalized_patch_entropy': patch_entropy.mean(axis=0),
        'mean_effective_patches': effective_patches.mean(axis=0),
        'mean_max_patch_probability': max_patch_probability.mean(axis=0),
    }
    for name, metric in aggregated.items():
        save_heatmap(metric, name.replace('_',' ').capitalize(), name.replace('_', ' '), outdir / f'{name}.png')
    if args.mean_spectrum:
        save_heatmap(spectral_slem, 'Second-largest eigenvalue modulus of mean attention matrix',
                     '|lambda_2| (of across-image mean P)', outdir / 'mean_transition_slem.png')

    # The average stationary patch map, not the stationary vector of average attention.
    for layer in range(n_layers):
        patch_grid = int(round(np.sqrt(n_tokens-1)))
        if patch_grid * patch_grid == n_tokens-1:
            layer_map = mean_patch_maps[layer].mean(axis=0).reshape(patch_grid, patch_grid)
            fig, ax = plt.subplots(figsize=(6, 5.5))
            im = ax.imshow(layer_map, cmap='viridis')
            ax.set_title(f'Mean stationary patch distribution: layer {layer:02d}\nmean over images and heads')
            ax.set_xlabel('Patch column')
            ax.set_ylabel('Patch row')
            fig.colorbar(im, ax=ax)
            fig.tight_layout()
            fig.savefig(outdir / f'mean_stationary_layer_{layer:02d}.png', dpi=180)
            plt.close(fig)

    with (outdir / 'layer_head_summary.csv').open('w', newline='') as fp:
        fieldnames = ['layer','head',*aggregated.keys(),'std_cls_mass','std_patch_entropy','max_stationary_residual','slem_of_mean_transition']
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        for layer in range(n_layers):
            for head in range(n_heads):
                writer.writerow({
                    'layer':layer, 'head':head,
                    **{k:float(v[layer,head]) for k,v in aggregated.items()},
                    'std_cls_mass':float(cls_mass[:,layer,head].std()),
                    'std_patch_entropy':float(patch_entropy[:,layer,head].std()),
                    'max_stationary_residual':float(residuals[:,layer,head].max()),
                    'slem_of_mean_transition':float(spectral_slem[layer,head])
                })

    np.savez_compressed(outdir / 'stationary_metrics_per_image.npz',
                        cls_mass=cls_mass, normalized_patch_entropy=patch_entropy,
                        effective_patches=effective_patches,
                        max_patch_probability=max_patch_probability,
                        stationary_residual=residuals,
                        mean_patch_maps=mean_patch_maps,
                        slem_of_mean_transition=spectral_slem)
    print('\nSaved aggregate results to:', outdir)
    print('Maximum fixed-point residual:', residuals.max())
    print('Interpretation: eigenvalue of MEAN attention != mean eigenvalue across images.')
    if residuals.max() > 1e-4:
        print('WARNING: Some stationary distributions did not converge accurately; inspect residuals and increase --max-iter.')

if __name__ == '__main__':
    main()
