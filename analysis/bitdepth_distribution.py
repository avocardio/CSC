"""Plot the per-channel bit-depth distribution at end of all-task training.

Tests the user's reframe: does CSC do "smart bit allocation" (some channels
at high precision, others freed up for future tasks) or just uniform compression?

Reads JSONs that include `compression.channel_bit_depths` (added 2026-04-28).
"""
from __future__ import annotations
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 10, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})


def load_bd_seeds(pattern: str) -> list[np.ndarray]:
    out = []
    for f in sorted(glob.glob(pattern)):
        d = json.load(open(f))
        c = d.get('compression', {})
        if 'channel_bit_depths' not in c:
            continue
        out.append(np.asarray(c['channel_bit_depths'], dtype=np.float32))
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--pattern', default='checkpoints/sup_cifar100_csc_t10_s*_bd.json')
    p.add_argument('--out', default='csc_paper/figures/bitdepth_distribution.pdf')
    p.add_argument('--init_bit', type=float, default=8.0)
    args = p.parse_args()

    seeds = load_bd_seeds(args.pattern)
    if not seeds:
        print(f'no JSONs matching {args.pattern}'); return

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))

    # Panel 1: histogram across all seeds
    all_bd = np.concatenate(seeds)
    ax1.hist(all_bd, bins=40, range=(0, args.init_bit), color='#1565C0',
             edgecolor='black', linewidth=0.6, alpha=0.85)
    ax1.axvline(all_bd.mean(), color='red', ls='--', lw=1.2,
                label=f'mean = {all_bd.mean():.2f} bits')
    ax1.axvline(args.init_bit, color='black', ls=':', lw=1.0,
                label=f'init = {args.init_bit:.0f} bits')
    ax1.set_xlabel('Per-channel bit-depth $b_i$')
    ax1.set_ylabel('Channel count')
    ax1.set_title(f'CSC bit-depth distribution at end of training '
                  f'({len(seeds)} seeds, {len(all_bd)} channels)')
    ax1.legend(loc='upper left', frameon=True)
    ax1.set_xlim(0, args.init_bit)

    # Panel 2: ECDF (cumulative) per seed, overlaid
    for k, bd in enumerate(seeds):
        x = np.sort(bd)
        y = np.arange(1, len(x) + 1) / len(x)
        ax2.plot(x, y, lw=1.4, alpha=0.85, label=f'seed {k}')
    ax2.set_xlabel('Bit-depth $b_i$')
    ax2.set_ylabel('Fraction of channels with $b \\le b_i$')
    ax2.set_title('Bit-depth ECDF per seed')
    ax2.set_xlim(0, args.init_bit)
    ax2.set_ylim(0, 1.02)
    ax2.grid(alpha=0.25)
    ax2.legend(loc='lower right', frameon=True)

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    fig.savefig(args.out)
    fig.savefig(args.out.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Wrote {args.out}')

    # Print summary stats
    print(f'\nSummary across {len(seeds)} seeds, {len(all_bd)} channels:')
    print(f'  mean       = {all_bd.mean():.3f}  (= {100*all_bd.mean()/args.init_bit:.1f}% of init)')
    print(f'  std        = {all_bd.std():.3f}')
    print(f'  Q05/50/95  = {np.percentile(all_bd, [5, 50, 95])}')
    print(f'  >= 7 bits  = {100*(all_bd>=7).mean():.1f}% of channels')
    print(f'  <= 2 bits  = {100*(all_bd<=2).mean():.1f}% of channels')
    print(f'  3-6 bits   = {100*((all_bd>2)&(all_bd<7)).mean():.1f}% of channels')


if __name__ == '__main__':
    main()
