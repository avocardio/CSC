"""Generate all final publication figures."""

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.dpi': 150,
    'font.family': 'serif',
})

os.makedirs('figures', exist_ok=True)

# =============================================================
# Figure A: Split CIFAR-100 scaling with DER++
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

# Data (mean ± std where available)
tasks = [10, 20, 50]
methods = {
    'Soft CSC + DER++ (r=200)': {
        'acc': [78.41, 83.70, 89.61],
        'std': [None, 0.47, None],
        'color': '#2196F3', 'marker': 'o',
    },
    'Plain DER++ (r=200)': {
        'acc': [77.91, 83.35, None],
        'std': [None, 0.95, None],
        'color': '#9C27B0', 'marker': 'D',
    },
    'Replay-only (r=200)': {
        'acc': [71.42, 77.51, 88.58],
        'std': [None, None, None],
        'color': '#4CAF50', 'marker': 's',
    },
    'PackNet': {
        'acc': [87.18, 87.38, 87.41],
        'std': [None, 0.37, None],
        'color': '#FF5722', 'marker': '^',
    },
}

for name, d in methods.items():
    valid = [(t, a, s) for t, a, s in zip(tasks, d['acc'], d['std']) if a is not None]
    ts = [v[0] for v in valid]
    accs = [v[1] for v in valid]
    stds = [v[2] if v[2] is not None else 0 for v in valid]
    ax.errorbar(ts, accs, yerr=stds, fmt=f"{d['marker']}-", color=d['color'],
                label=name, markersize=8, linewidth=2, capsize=4)

ax.set_xlabel('Number of Tasks')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xticks(tasks)
ax.legend(loc='lower left')
ax.grid(True, alpha=0.3)
ax.set_ylim(65, 95)

plt.tight_layout()
plt.savefig('figures/cifar100_scaling_der.pdf', bbox_inches='tight')
print("Saved figures/cifar100_scaling_der.pdf")

# =============================================================
# Figure B: Permuted MNIST scaling
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5.5))

pmnist_tasks = [10, 20, 50, 100]
pmnist = {
    'Soft CSC': {
        'acc': [90.45, 83.89, 74.91, 69.45],
        'color': '#2196F3', 'marker': 'o',
    },
    'Replay-only': {
        'acc': [88.91, 83.06, 76.80, 73.47],
        'color': '#4CAF50', 'marker': 's',
    },
    'PackNet': {
        'acc': [95.18, 66.87, 24.28, 18.77],
        'color': '#FF5722', 'marker': '^',
    },
}

for name, d in pmnist.items():
    ax.plot(pmnist_tasks, d['acc'], f"{d['marker']}-", color=d['color'],
            label=name, markersize=8, linewidth=2)

ax.set_xlabel('Number of Tasks')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xscale('log')
ax.set_xticks(pmnist_tasks)
ax.set_xticklabels(pmnist_tasks)
ax.legend()
ax.grid(True, alpha=0.3)
ax.set_ylim(0, 100)

plt.tight_layout()
plt.savefig('figures/pmnist_scaling.pdf', bbox_inches='tight')
print("Saved figures/pmnist_scaling.pdf")

# =============================================================
# Figure C: LR Control
# =============================================================
fig, ax = plt.subplots(figsize=(7, 4.5))

lr_labels = ['1e-3\n(base)', '3.3e-4\n(/3)', '1.1e-4\n(/9)', '6.7e-5\n(/15)']
lr_accs = [71.42, 74.49, 72.88, 69.42]
soft_csc_best = 76.66

x = np.arange(len(lr_labels))
ax.bar(x, lr_accs, color='#4CAF50', alpha=0.7, width=0.6, label='Replay-only (tuned LR)')
ax.axhline(y=soft_csc_best, color='#2196F3', linewidth=2.5, linestyle='--',
           label=f'Soft CSC ({soft_csc_best}%)')
ax.set_xticks(x)
ax.set_xticklabels(lr_labels)
ax.set_ylabel('Average Accuracy (%)')
ax.set_xlabel('Learning Rate')
ax.legend()
ax.set_ylim(65, 80)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/lr_control.pdf', bbox_inches='tight')
print("Saved figures/lr_control.pdf")

# =============================================================
# Figure D: Pareto frontier (accuracy vs bits)
# =============================================================
fig, ax = plt.subplots(figsize=(8, 5))

pareto = [
    (100, 71.42, 'Replay r=200', '#4CAF50', 's'),
    (100, 76.03, 'Replay r=500', '#4CAF50', 's'),
    (24.5, 76.12, 'Soft CSC g=0.001', '#2196F3', 'o'),
    (22.2, 76.77, 'Soft CSC g=0.005', '#2196F3', 'o'),
    (19.1, 76.66, 'Soft CSC g=0.01', '#2196F3', 'o'),
    (8.7, 69.67, 'CSC g=0.01 (no prot)', '#90CAF9', 'o'),
    (20.5, 71.00, 'CSC g=0.001 (no prot)', '#90CAF9', 'o'),
    (24.5, 78.41, 'CSC+DER++ r=200', '#E91E63', 'D'),
    (24.5, 82.98, 'CSC+DER++ r=500', '#E91E63', 'D'),
    (94.4, 87.18, 'PackNet', '#FF5722', '^'),
]

for bits, acc, label, color, marker in pareto:
    ax.scatter(bits, acc, color=color, s=80, marker=marker, zorder=5)
    offset = (5, 5) if bits < 90 else (-60, 5)
    ax.annotate(label, (bits, acc), fontsize=7, xytext=offset,
                textcoords='offset points')

ax.set_xlabel('Model Size (% of original bits)')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xlim(-5, 110)
ax.set_ylim(65, 92)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/pareto_frontier.pdf', bbox_inches='tight')
print("Saved figures/pareto_frontier.pdf")

print("\nAll figures generated!")
