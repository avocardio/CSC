"""Model-size scaling plot: CSC's capacity-efficiency advantage as a function
of model parameters.

For each (model architecture, num_params) point, plots final average accuracy
on Split CIFAR-100 (10 tasks). Solid line = CSC, dashed = DER++, dotted = ER,
gray = Finetune. Capacity used (% bits) is annotated next to CSC points.

Auto-detects available models from JSONs.

Usage: python analysis/scaling_plot.py --ckpt_dir /tmp/cluster_jsons
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

# Approximate parameter counts (M) for the scaling axis.
PARAMS_M = {
    'mlp':       0.27,
    'vit_tiny':  5.4,
    'resnet18':  11.2,
    'resnet50':  23.7,
    'vit_small': 21.4,
}
# Display labels for the x-axis (instead of just numeric)
MODEL_LABELS = {
    'mlp':       'MLP\n0.3M',
    'vit_tiny':  'ViT-Tiny\n5.4M',
    'resnet18':  'ResNet-18\n11M',
    'vit_small': 'ViT-Small\n21M',
    'resnet50':  'ResNet-50\n24M',
}

METHOD_STYLE = {
    'finetune': {'label': 'Finetune', 'color': '#888888', 'marker': 'x', 'ls': ':'},
    'replay':   {'label': 'ER',       'color': '#4CAF50', 'marker': 's', 'ls': '--'},
    'der':      {'label': 'DER++',    'color': '#9C27B0', 'marker': 'D', 'ls': '--'},
    'csc':      {'label': 'CSC (ours)', 'color': '#1565C0', 'marker': 'o', 'ls': '-'},
}


def load(ckpt_dir: str, dataset: str, num_tasks: int) -> dict:
    by = defaultdict(list)
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'sup_*.json'))):
        d = json.load(open(f))
        cfg = d['config']
        if cfg.get('dataset', 'cifar100') != dataset: continue
        if cfg.get('num_tasks') != num_tasks: continue
        model = cfg.get('model', 'resnet18')
        method = cfg.get('method')
        by[(model, method)].append({
            'acc': d['final_avg'] * 100,
            'bits': (d.get('compression') or {}).get('compression_ratio', 1.0) * 100,
        })
    return by


def plot(out_pdf: str, by: dict, num_tasks: int):
    fig, ax = plt.subplots(figsize=(7.2, 4.6))
    models = sorted(set(k[0] for k in by), key=lambda m: PARAMS_M.get(m, 1.0))
    for method in METHOD_STYLE:
        s = METHOD_STYLE[method]
        xs, ys, errs, bits_list = [], [], [], []
        for model in models:
            rows = by.get((model, method))
            if not rows:
                continue
            accs = np.array([r['acc'] for r in rows])
            bits = np.array([r['bits'] for r in rows])
            xs.append(PARAMS_M.get(model, np.nan))
            ys.append(accs.mean())
            errs.append(accs.std() if len(accs) > 1 else 0.0)
            bits_list.append(bits.mean())
        if not xs:
            continue
        xs = np.array(xs); ys = np.array(ys); errs = np.array(errs)
        ax.errorbar(xs, ys, yerr=errs, color=s['color'], ls=s['ls'],
                    lw=2.2 if method == 'csc' else 1.4,
                    marker=s['marker'], markersize=9 if method == 'csc' else 6,
                    markeredgecolor='black', markeredgewidth=0.6,
                    label=s['label'], capsize=3, alpha=0.95)
        if method == 'csc':
            for x, y, b in zip(xs, ys, bits_list):
                ax.annotate(f'{b:.0f}%', xy=(x, y),
                            xytext=(0, -18), textcoords='offset points',
                            fontsize=8, color=s['color'], ha='center',
                            fontweight='bold')

    ax.set_xscale('log')
    ax.set_xlabel('Model parameters')
    ax.set_ylabel('Final avg accuracy (%)')
    # Custom x-tick labels (clear minor ticks first to avoid overlap with log defaults)
    tick_models = [m for m in models]
    tick_x = [PARAMS_M[m] for m in tick_models]
    tick_l = [MODEL_LABELS.get(m, f'{m}\n{PARAMS_M[m]:.1f}M') for m in tick_models]
    ax.set_xticks(tick_x, minor=False)
    ax.set_xticks([], minor=True)
    ax.set_xticklabels(tick_l, minor=False)
    from matplotlib.ticker import NullFormatter
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_title(f'Model-size scaling on {num_tasks}-task Split CIFAR-100')
    ax.grid(alpha=0.25, which='major')
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
    p.add_argument('--dataset', default='cifar100', choices=['cifar100'])
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    by = load(args.ckpt_dir, args.dataset, args.num_tasks)
    print(f'Coverage:')
    for k in sorted(by):
        print(f'  {k}: n={len(by[k])}')
    plot(os.path.join(args.out_dir, 'scaling.pdf'), by, args.num_tasks)


if __name__ == '__main__':
    main()
