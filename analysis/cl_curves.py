"""Two canonical CL plots used in most papers:

(1) running_avg.pdf  — running average accuracy across tasks seen so far,
                       plotted vs task index. One line per method, error band
                       across seeds. THE canonical CL learning curve.

(2) bwt_vs_acc.pdf   — forgetting (BWT) on x-axis, average accuracy on y-axis,
                       marker size encodes capacity used (smaller = more
                       compressed). Replaces the Pareto plot, which became a
                       trivial vertical line once all baselines clustered at
                       100% bits.

Plus the existing heatmaps and tables.

Usage:
  python analysis/cl_curves.py --ckpt_dir checkpoints --num_tasks 10 --dataset cifar100
"""
from __future__ import annotations
import os, sys, json, glob, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 11,
    'axes.labelsize': 12, 'axes.titlesize': 12,
    'legend.fontsize': 9, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

METHOD_STYLE = {
    'finetune': {'label': 'Finetune', 'color': '#888888', 'marker': 'x', 'lw': 1.4, 'ls': '--'},
    'ewc':      {'label': 'EWC',      'color': '#795548', 'marker': 'v', 'lw': 1.4, 'ls': '-'},
    'replay':   {'label': 'ER',       'color': '#4CAF50', 'marker': 's', 'lw': 1.6, 'ls': '-'},
    'der':      {'label': 'DER++',    'color': '#9C27B0', 'marker': 'D', 'lw': 1.6, 'ls': '-'},
    'csc':      {'label': 'CSC (ours)', 'color': '#1565C0', 'marker': 'o', 'lw': 2.2, 'ls': '-'},
}


def load_matrices(ckpt_dir: str, num_tasks: int, dataset: str,
                  model: str | None = None) -> dict:
    """Filter by (dataset, num_tasks, model). Default model = resnet18 for
    cifar100, mlp for pmnist; skip tag'd ablation runs."""
    if model is None:
        model = 'mlp' if dataset == 'pmnist' else 'resnet18'
    by_method: dict[str, list] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'sup_*.json'))):
        d = json.load(open(f))
        cfg = d['config']
        if cfg.get('dataset', 'cifar100') != dataset:
            continue
        if cfg.get('num_tasks') != num_tasks:
            continue
        if cfg.get('model', 'resnet18') != model:
            continue
        if cfg.get('tag', ''):
            continue
        by_method[cfg['method']].append(np.array(d['accuracy_matrix']) * 100)
    return {m: np.stack(ms) for m, ms in by_method.items() if ms}


def load_aggregates(ckpt_dir: str, num_tasks: int, dataset: str,
                    model: str | None = None) -> dict:
    if model is None:
        model = 'mlp' if dataset == 'pmnist' else 'resnet18'
    by: dict[str, list] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'sup_*.json'))):
        d = json.load(open(f))
        cfg = d['config']
        if cfg.get('dataset', 'cifar100') != dataset or cfg.get('num_tasks') != num_tasks:
            continue
        if cfg.get('model', 'resnet18') != model:
            continue
        if cfg.get('tag', ''):
            continue
        by[cfg['method']].append({
            'acc': d['final_avg'] * 100,
            'bwt': d['bwt'] * 100,
            'bits': (d.get('compression') or {}).get('compression_ratio', 1.0) * 100,
        })
    return by


def plot_running_avg(matrices: dict, num_tasks: int, aggregates: dict,
                     out_pdf: str, title: str):
    """Mean accuracy across all tasks seen so far, vs task index. One line per method.
    Capacity used annotated in the legend; markers only at endpoints to reduce noise."""
    fig, ax = plt.subplots(figsize=(6.5, 4.2))
    methods_sorted = [m for m in METHOD_STYLE if m in matrices]
    for method in methods_sorted:
        mats = matrices[method]                          # (n_seeds, T, T)
        T = mats.shape[1]
        per_seed = np.zeros((mats.shape[0], T))
        for k in range(T):
            per_seed[:, k] = mats[:, k, : k + 1].mean(axis=1)
        mean = per_seed.mean(axis=0)
        std = per_seed.std(axis=0) if mats.shape[0] > 1 else np.zeros_like(mean)
        x = np.arange(1, T + 1)
        s = METHOD_STYLE[method]
        # Capacity tag in legend label (the whole point of the paper)
        bits = np.mean([r['bits'] for r in aggregates.get(method, [{'bits': 100}])])
        cap_tag = f' ({bits:.0f}%)'
        ax.plot(x, mean, color=s['color'], lw=s['lw'], ls=s['ls'],
                label=f"{s['label']}{cap_tag}")
        # Endpoint marker only (less noise than markers on every point)
        ax.scatter([x[-1]], [mean[-1]], s=60, color=s['color'], marker=s['marker'],
                   edgecolor='black', linewidth=0.5, zorder=5)
        if mats.shape[0] > 1:
            ax.fill_between(x, mean - std, mean + std, color=s['color'], alpha=0.13, lw=0)
    ax.set_xlabel('Tasks seen')
    ax.set_ylabel('Mean accuracy across seen tasks (%)')
    ax.set_title(title)
    ax.set_xticks(np.arange(1, num_tasks + 1, max(1, num_tasks // 10)))
    ax.set_xlim(0.6, num_tasks + 0.4)
    # Clip y-axis just below the lowest mean (so the early-task spread is readable)
    all_means = np.concatenate([
        np.array([matrices[m][s, k, : k + 1].mean()
                  for s in range(matrices[m].shape[0]) for k in range(num_tasks)])
        for m in methods_sorted])
    ymin = max(0, np.percentile(all_means, 1) - 5)
    ax.set_ylim(ymin, 100)
    ax.grid(alpha=0.25)
    ax.legend(loc='lower right', frameon=True, title='method (capacity)')
    plt.tight_layout()
    fig.savefig(out_pdf); fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Wrote: {out_pdf}')


def plot_bwt_vs_acc(aggregates: dict, num_tasks: int, out_pdf: str, title: str,
                    drop_finetune: bool = True):
    """Forgetting (BWT) on x, accuracy on y, marker SIZE = capacity used.
    Larger marker = more capacity (worse efficiency). CSC has a small marker
    in the upper-right = high acc + low forgetting + small capacity = winner.

    By default drops the Finetune outlier (caption mentions it instead) so
    the interesting region is not squashed."""
    fig, ax = plt.subplots(figsize=(6.4, 4.4))
    for method in METHOD_STYLE:
        if method not in aggregates:
            continue
        if drop_finetune and method == 'finetune':
            continue
        rows = aggregates[method]
        accs = np.array([r['acc'] for r in rows])
        bwts = np.array([r['bwt'] for r in rows])
        bits = np.array([r['bits'] for r in rows])
        s = METHOD_STYLE[method]
        # Marker size proportional to capacity (100% -> ~480, 20% -> ~96)
        marker_size = 60 + 4.2 * bits.mean()
        ax.errorbar(bwts.mean(), accs.mean(),
                    xerr=bwts.std(), yerr=accs.std(),
                    fmt='none', ecolor=s['color'], capsize=4, lw=1.2, alpha=0.6)
        ax.scatter([bwts.mean()], [accs.mean()],
                   s=marker_size, color=s['color'], marker=s['marker'],
                   edgecolor='black', linewidth=0.9, zorder=5, alpha=0.9,
                   label=f"{s['label']}: {bits.mean():.0f}% capacity")
        # Annotate accuracy outside the marker, far enough to clear it.
        # CSC (the smallest marker = our method) goes ABOVE so it doesn't
        # collide with the larger DER++ diamond directly below.
        radius_pts = 0.5 * (marker_size ** 0.5)              # marker s is area
        if method == 'csc':
            ax.annotate(f'{accs.mean():.1f}%',
                        xy=(bwts.mean(), accs.mean()),
                        xytext=(0, radius_pts + 4), textcoords='offset points',
                        fontsize=9, color=s['color'], fontweight='bold',
                        ha='center', va='bottom')
        else:
            ax.annotate(f'{accs.mean():.1f}%',
                        xy=(bwts.mean(), accs.mean()),
                        xytext=(0, -(radius_pts + 4)), textcoords='offset points',
                        fontsize=9, color=s['color'], fontweight='bold',
                        ha='center', va='top')

    ax.set_xlabel('Backward transfer (closer to 0 = less forgetting)')
    ax.set_ylabel('Final average accuracy (%)')
    ax.set_title(title)
    ax.grid(alpha=0.25)
    # Pad the right side so accuracy annotations don't clip
    xlim = ax.get_xlim()
    ax.set_xlim(xlim[0], xlim[1] + 0.15 * (xlim[1] - xlim[0]))
    note = ('Finetune dropped (BWT $\\approx$ -60); '
            if drop_finetune else '')
    ax.text(0.02, 0.02, f'{note}marker size $\\propto$ capacity',
            transform=ax.transAxes, fontsize=8, color='#555555',
            verticalalignment='bottom')
    ax.legend(loc='lower right', frameon=True)
    plt.tight_layout()
    fig.savefig(out_pdf); fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Wrote: {out_pdf}')


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', default='checkpoints')
    p.add_argument('--out_dir', default='csc_paper/figures')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--dataset', default='cifar100', choices=['cifar100', 'pmnist'])
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = '' if args.dataset == 'cifar100' else f'_{args.dataset}'
    title_suffix = ', Split CIFAR-100' if args.dataset == 'cifar100' else ', Permuted MNIST'
    title = f'{args.num_tasks} tasks{title_suffix}'

    matrices = load_matrices(args.ckpt_dir, args.num_tasks, args.dataset)
    aggregates = load_aggregates(args.ckpt_dir, args.num_tasks, args.dataset)
    print(f'Methods: {list(matrices)}')

    plot_running_avg(matrices, args.num_tasks, aggregates,
                     os.path.join(args.out_dir, f'running_avg{suffix}.pdf'),
                     title)
    plot_bwt_vs_acc(aggregates, args.num_tasks,
                    os.path.join(args.out_dir, f'bwt_vs_acc{suffix}.pdf'),
                    title)


if __name__ == '__main__':
    main()
