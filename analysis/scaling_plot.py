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
    ax.set_xlabel('')                                            # labels carry the meaning
    ax.set_ylabel('Final avg accuracy (%)')
    # Hide default log tick labels and place our own as text (staggered when
    # adjacent labels would overlap on a log scale).
    from matplotlib.ticker import NullFormatter
    ax.set_xticks([])
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.xaxis.set_major_formatter(NullFormatter())
    # Pad x-range so labels at the edges have room
    xs_all = sorted(PARAMS_M[m] for m in models)
    if xs_all:
        ax.set_xlim(xs_all[0] / 1.4, xs_all[-1] * 1.4)
    ymin, _ = ax.get_ylim()
    # Place labels below the x-axis. Stagger when consecutive labels
    # are within < 1.3x of each other on log scale.
    sorted_models = sorted(models, key=lambda m: PARAMS_M.get(m, 0))
    last_x = 0
    stagger = False
    for m in sorted_models:
        x = PARAMS_M[m]
        label = MODEL_LABELS.get(m, f'{m}\n{x:.1f}M')
        if last_x > 0 and (x / last_x) < 1.3:
            stagger = not stagger
        else:
            stagger = False
        last_x = x
        y_text = -0.05 if not stagger else -0.14
        ax.text(x, y_text, label, transform=ax.get_xaxis_transform(),
                ha='center', va='top', fontsize=10)
        ax.axvline(x, color='#cccccc', lw=0.5, zorder=0)
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
