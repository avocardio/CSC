"""Per-task accuracy curves and learning matrix from JSON outputs.

For each method × seed, plot the standard CL "accuracy of task j after training
on task i" curves. Two views:

  per_task_decay.pdf  — for each task j, accuracy across training-positions
                        i = j, j+1, ..., T-1. Shows forgetting trajectory.
  accuracy_matrix.pdf — heatmap of A[i][j], one panel per method (seed-mean).

Usage: python analysis/per_task_curves.py --ckpt_dir checkpoints --num_tasks 10
"""
from __future__ import annotations
import os, sys, json, argparse, glob
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({
    'font.family': 'serif', 'font.size': 10,
    'axes.labelsize': 11, 'axes.titlesize': 11,
    'legend.fontsize': 8, 'figure.dpi': 150,
    'axes.spines.top': False, 'axes.spines.right': False,
})

METHOD_STYLE = {
    'finetune': {'label': 'Finetune', 'color': '#888888'},
    'replay':   {'label': 'ER',       'color': '#4CAF50'},
    'der':      {'label': 'DER++',    'color': '#9C27B0'},
    'ewc':      {'label': 'EWC',      'color': '#795548'},
    'csc':      {'label': 'CSC',      'color': '#1565C0'},
}


def load_matrices(ckpt_dir: str, num_tasks: int, dataset: str = 'cifar100') -> dict:
    """Group A[i][j] matrices by method (filtered to one dataset), seed-averaged."""
    by_method: dict[str, list] = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'sup_*.json'))):
        d = json.load(open(f))
        cfg = d['config']
        ds = cfg.get('dataset', 'cifar100')
        if ds != dataset or cfg.get('num_tasks') != num_tasks:
            continue
        method = cfg.get('method')
        mat = np.array(d['accuracy_matrix']) * 100
        by_method[method].append(mat)
    return {m: np.stack(ms) for m, ms in by_method.items() if ms}


def plot_per_task_decay(matrices: dict, num_tasks: int, out_pdf: str):
    """For each task j, plot mean accuracy across training-positions i ≥ j."""
    methods = [m for m in METHOD_STYLE if m in matrices]
    if not methods:
        print('  no methods to plot'); return
    n_panels = num_tasks
    cols = min(5, n_panels)
    rows = (n_panels + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(2.4 * cols, 2.0 * rows),
                             sharey=True)
    axes = axes.flatten() if hasattr(axes, 'flatten') else [axes]
    for j in range(num_tasks):
        ax = axes[j]
        for method in methods:
            mats = matrices[method]                # (n_seeds, T, T)
            seed_curves = mats[:, j:, j]           # (n_seeds, T - j) for this task
            mean = seed_curves.mean(axis=0)
            std = seed_curves.std(axis=0) if seed_curves.shape[0] > 1 else np.zeros_like(mean)
            xs = np.arange(j, num_tasks)
            style = METHOD_STYLE[method]
            ax.plot(xs, mean, color=style['color'], lw=1.5, label=style['label'])
            if seed_curves.shape[0] > 1:
                ax.fill_between(xs, mean - std, mean + std, alpha=0.18,
                                color=style['color'], lw=0)
        ax.set_title(f'Task {j}', fontsize=9, pad=2)
        ax.set_xticks(np.arange(j, num_tasks, max(1, num_tasks // 5)))
        ax.set_xlim(j - 0.5, num_tasks - 0.5)
        ax.set_ylim(0, 100)
        ax.grid(alpha=0.2)
    for k in range(n_panels, len(axes)):
        axes[k].axis('off')
    fig.text(0.5, 0.02, 'Training position $i$ (task being trained on)', ha='center')
    fig.text(0.005, 0.5, 'Accuracy on task $j$ (%)', va='center', rotation='vertical')
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper center', ncol=len(methods), frameon=False,
               bbox_to_anchor=(0.5, 1.0))
    plt.tight_layout(rect=(0.02, 0.04, 1, 0.95))
    fig.savefig(out_pdf); fig.savefig(out_pdf.replace('.pdf', '.png'), dpi=180)
    plt.close(fig)
    print(f'Wrote: {out_pdf}')


def plot_matrix_heatmaps(matrices: dict, num_tasks: int, out_pdf: str):
    """One heatmap per method showing the lower-triangular accuracy matrix."""
    methods = [m for m in METHOD_STYLE if m in matrices]
    if not methods:
        return
    fig, axes = plt.subplots(1, len(methods), figsize=(2.2 * len(methods), 2.4),
                             squeeze=False)
    axes = axes[0]
    for ax, method in zip(axes, methods):
        mats = matrices[method].mean(axis=0)
        # Mask upper triangle (above diagonal: model hasn't seen task j yet)
        mask = np.triu(np.ones_like(mats), k=1)
        masked = np.ma.masked_array(mats, mask=mask.astype(bool))
        im = ax.imshow(masked, cmap='viridis', vmin=0, vmax=100, aspect='equal')
        ax.set_title(METHOD_STYLE[method]['label'], fontsize=10)
        ax.set_xlabel('eval task $j$')
        ax.set_xticks(range(num_tasks))
        ax.set_yticks(range(num_tasks))
        if method == methods[0]:
            ax.set_ylabel('after training task $i$')
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8, label='Accuracy (%)')
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
    matrices = load_matrices(args.ckpt_dir, args.num_tasks, dataset=args.dataset)
    print(f'Methods with data: {list(matrices)}')
    suffix = '' if args.dataset == 'cifar100' else f'_{args.dataset}'
    plot_per_task_decay(matrices, args.num_tasks,
                        os.path.join(args.out_dir, f'per_task_decay{suffix}.pdf'))
    plot_matrix_heatmaps(matrices, args.num_tasks,
                         os.path.join(args.out_dir, f'accuracy_matrix{suffix}.pdf'))


if __name__ == '__main__':
    main()
