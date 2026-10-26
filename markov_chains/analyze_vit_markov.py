#!/usr/bin/env python3
"""Inspect Markov-chain eigenvalues and stationary distribution for one ViT attention head.

Example:
    python analyze_vit_markov.py --image 97 --layer 5 --head 3

Input is the (118, 12, 12, 197, 197) attention .npy produced by
extract_vit_attention.py. Generates PNG figures and a concise report.
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.linalg import eig

DEFAULT_INPUT = Path('/home/maria/ViTExplainabilityExperimental/data/vit_attention/vit_attention_float16.npy')


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--input', type=Path, default=DEFAULT_INPUT)
    ap.add_argument('--image', type=int, default=97)
    ap.add_argument('--layer', type=int, default=5)
    ap.add_argument('--head', type=int, default=3)
    ap.add_argument('--out', type=Path, default=None)
    args = ap.parse_args()
    A = np.load(args.input, mmap_mode='r')
    if A.ndim != 5:
        raise ValueError(f'Expected 5D array; got shape {A.shape}')
    n_images, n_layers, n_heads, n_tokens, other_tokens = A.shape
    if n_tokens != other_tokens:
        raise ValueError('Attention matrices must be square')
    for label, index, bound in [('image', args.image, n_images), ('layer', args.layer, n_layers), ('head', args.head, n_heads)]:
        if not 0 <= index < bound:
            ap.error(f'--{label} must be 0...{bound-1}')
    out = args.out or args.input.parent / 'markov_analysis'
    out.mkdir(parents=True, exist_ok=True)
    stem = f'scene_{args.image:03d}_layer_{args.layer:02d}_head_{args.head:02d}'

    # Float16 truncation can disturb exact stochasticity. Repair it first.
    P = np.array(A[args.image, args.layer, args.head], dtype=np.float64)
    if not np.isfinite(P).all() or np.any(P < 0):
        raise ValueError('Matrix contains nonfinite or negative entries')
    row_sums = P.sum(axis=1)
    if np.any(row_sums <= 0):
        raise ValueError('A row of attention has zero mass')
    raw_error = float(np.max(np.abs(row_sums - 1.0)))
    P /= row_sums[:, None]

    # For row-stochastic P: pi @ P = pi, so pi is a LEFT eigenvector.
    # scipy.linalg.eig returns left eigenvectors if left=True.
    vals, left_vecs = eig(P, left=True, right=False)
    order = np.argsort(-np.abs(vals))
    vals = vals[order]
    left_vecs = left_vecs[:, order]
    one = np.argmin(np.abs(vals - 1.0))
    pi = np.real(left_vecs[:, one])
    if pi.sum() < 0:
        pi *= -1
    # Numerical eigenvector entries can have tiny negative roundoff.
    if pi.min() < -1e-9:
        raise RuntimeError('Unexpected negative stationary eigenvector: check reducibility/numerics')
    pi = np.maximum(pi, 0)
    pi /= pi.sum()
    residual = float(np.max(np.abs(pi @ P - pi)))

    print(f'Input: {args.input}')
    print(f'Matrix: image {args.image}, layer {args.layer}, head {args.head}, shape {P.shape}')
    print(f'Raw float16 row-sum max error: {raw_error:.3g}; normalized row-sum max error: {np.max(np.abs(P.sum(axis=1) - 1)):.3g}')
    print(f'Closest eigenvalue to 1: {vals[one]}')
    print(f'Stationarity residual max |pi P - pi|: {residual:.3g}')
    print(f'Stationary CLS token probability pi[0]: {pi[0]:.6f}')
    print('Top 10 eigenvalues by absolute magnitude:')
    for i, value in enumerate(vals[:10], start=1):
        print(f'  {i:2d}: {value.real:+.6f}{value.imag:+.6f}j  |lambda|={abs(value):.6f}')

    # Eigenspectrum in complex plane (non-symmetric stochastic matrices can have complex eigenvalues).
    fig, ax = plt.subplots(figsize=(6, 6))
    theta = np.linspace(0, 2 * np.pi, 400)
    ax.plot(np.cos(theta), np.sin(theta), '--', linewidth=1, label='Unit circle')
    ax.scatter(vals.real, vals.imag, s=15, alpha=.75)
    ax.scatter([1], [0], marker='x', s=90, label='Eigenvalue 1')
    ax.axhline(0, linewidth=.5)
    ax.axvline(0, linewidth=.5)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xlabel('Real part')
    ax.set_ylabel('Imaginary part')
    ax.set_title(f'Attention Markov eigenspectrum: {stem}')
    ax.legend()
    fig.tight_layout()
    eig_path = out / f'{stem}_eigenvalues.png'
    fig.savefig(eig_path, dpi=170)
    plt.close(fig)

    # Stationary distribution over patches (CLS is a separate state).
    n_patch = n_tokens - 1
    side = int(round(n_patch ** .5))
    if side * side != n_patch:
        raise ValueError(f'Expected square patch grid, got {n_patch} patches')
    patch_map = pi[1:].reshape(side, side)
    fig, ax = plt.subplots(figsize=(6, 5))
    img = ax.imshow(patch_map, interpolation='nearest')
    fig.colorbar(img, ax=ax, label='Stationary probability per patch')
    ax.set_xlabel('Patch column')
    ax.set_ylabel('Patch row')
    ax.set_title(f'Stationary distribution (CLS mass={pi[0]:.3f})\n{stem}')
    fig.tight_layout()
    stat_path = out / f'{stem}_stationary.png'
    fig.savefig(stat_path, dpi=170)
    plt.close(fig)

    # To see actual mixing, begin at CLS and repeatedly multiply by P.
    steps = [0, 1, 2, 5, 10, 25, 50, 100]
    p = np.zeros(n_tokens)
    p[0] = 1.0
    rows = []
    for t in range(max(steps) + 1):
        if t in steps:
            rows.append((t, p.copy(), float(np.abs(p - pi).sum() / 2)))
        p = p @ P
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot([t for t, _, _ in rows], [tv for _, _, tv in rows], 'o-')
    ax.set_xlabel('Number of transitions')
    ax.set_ylabel('Total variation distance to stationarity')
    ax.set_title(f'Mixing from the CLS token: {stem}')
    fig.tight_layout()
    mix_path = out / f'{stem}_mixing.png'
    fig.savefig(mix_path, dpi=170)
    plt.close(fig)

    np.savez(out / f'{stem}_results.npz', eigenvalues=vals, stationary=pi,
             mixing_steps=np.array([t for t, _, _ in rows]),
             mixing_tv=np.array([tv for _, _, tv in rows]))
    print('\nSaved:')
    for path in [eig_path, stat_path, mix_path, out / f'{stem}_results.npz']:
        print(' ', path)


if __name__ == '__main__':
    main()
