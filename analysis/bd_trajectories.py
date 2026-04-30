"""Plot per-channel bit-depth trajectories across tasks.

Tests the user's hypothesis: do channels SPECIALIZE on specific tasks
(rise sharply on task k) and get REUSED later (drop after task k, rise
again on task k+1 with different specialization)?

Or do channels stay flat across tasks (no reallocation)?

Usage:  python analysis/bd_trajectories.py --json checkpoints/<run>.json
"""
from __future__ import annotations
import os, sys, json, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--json', required=True)
    p.add_argument('--out', default='csc_paper/figures/bd_trajectories.pdf')
    p.add_argument('--top_n', type=int, default=30,
                   help='Plot the N channels with highest variance across tasks (most interesting).')
    args = p.parse_args()

    d = json.load(open(args.json))
    traj = d.get('bd_trajectories')
    if not traj:
        print('No bd_trajectories field; nothing to plot.'); return

    T = len(traj)
    print(f'{T} task snapshots')
    layers = list(traj[0].keys())
    print(f'{len(layers)} quantized layers')

    # Concatenate all channels across all layers into one (T, N_channels) matrix.
    channels = []  # list of (layer, idx, traj_array)
    for layer in layers:
        for i in range(len(traj[0][layer])):
            arr = np.array([traj[t][layer][i] for t in range(T)])
            channels.append((layer, i, arr))
    n_total = len(channels)
    print(f'{n_total} channels total')

    # Per-channel summary stats
    mat = np.stack([c[2] for c in channels])  # (n_total, T)
    means = mat.mean(axis=1)
    stds = mat.std(axis=1)
    print(f'\nPer-channel summary:')
    print(f'  bd mean over channels: {means.mean():.2f} (channel-mean averaged)')
    print(f'  bd std-across-tasks: mean={stds.mean():.2f}  median={np.median(stds):.2f}  max={stds.max():.2f}')
    print(f'  channels with std > 0.5 (notable change across tasks): {int((stds > 0.5).sum())} / {n_total}')
    print(f'  channels with std > 1.0: {int((stds > 1.0).sum())} / {n_total}')
    print(f'  channels with std > 2.0: {int((stds > 2.0).sum())} / {n_total}')
    print(f'  channels with non-monotone trajectories (drop then rise): see plot')

    # Detect non-monotone (specialist + reuse) trajectories
    nonmonotone_count = 0
    for c in channels:
        a = c[2]
        # detect at least one drop of >0.5 followed by rise of >0.5
        diffs = np.diff(a)
        drops = np.where(diffs < -0.5)[0]
        rises = np.where(diffs > 0.5)[0]
        if len(drops) and len(rises) and rises.max() > drops.min():
            nonmonotone_count += 1
    print(f'  non-monotone (drop>=0.5 then rise>=0.5): {nonmonotone_count} / {n_total}')

    # Plot top-N most-variable channels
    top_idx = np.argsort(stds)[-args.top_n:][::-1]
    fig, axes = plt.subplots(2, 1, figsize=(8, 7))

    # Panel 1: top-N variable channels
    ax = axes[0]
    for idx in top_idx:
        layer, i, a = channels[idx]
        ax.plot(range(T), a, lw=0.8, alpha=0.5)
    ax.set_xlabel('task')
    ax.set_ylabel('bit-depth')
    ax.set_title(f'Top-{args.top_n} most-variable channels across tasks')
    ax.grid(alpha=0.3)
    ax.set_xticks(range(T))

    # Panel 2: histogram of per-channel std-across-tasks
    ax = axes[1]
    ax.hist(stds, bins=50, color='#1565C0', edgecolor='black', linewidth=0.4)
    ax.axvline(0.5, color='red', ls='--', lw=1, label='std=0.5')
    ax.axvline(1.0, color='red', ls='-', lw=1, label='std=1.0')
    ax.set_xlabel('std of bit-depth across tasks (per channel)')
    ax.set_ylabel('# channels')
    ax.set_title('Channel-wise variance: are channels static or specialising?')
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'\nWrote: {args.out}')


if __name__ == '__main__':
    main()
