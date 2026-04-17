"""Generate the two-panel scaling figure for the paper.

Left panel: Permuted MNIST scaling (10, 20, 50 tasks)
Right panel: Split CIFAR-100 scaling (10, 20, 25, 50 tasks)
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 11, 'figure.dpi': 150,
})

# Data from experiments
pmnist = {
    'tasks': [10, 20, 50],
    'soft_csc': [90.45, 83.89, 74.91],
    'replay': [88.91, 83.06, 76.80],
    'packnet': [95.18, 66.87, 24.28],
}

cifar = {
    'tasks': [10, 20, 25, 50],
    'soft_csc': [76.12, 81.42, 84.24, 90.35],
    'replay': [71.42, 77.51, 83.55, 88.58],
    'packnet': [87.18, 86.90, 87.87, 87.41],
}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))

# Left: Permuted MNIST
for method, marker, color in [
    ('soft_csc', 'o', '#2196F3'), ('replay', 's', '#4CAF50'), ('packnet', '^', '#FF5722')]:
    label = {'soft_csc': 'Soft CSC', 'replay': 'Replay-only', 'packnet': 'PackNet'}[method]
    ax1.plot(pmnist['tasks'], pmnist[method], f'{marker}-', color=color,
             label=label, markersize=8, linewidth=2)

ax1.set_xlabel('Number of Tasks')
ax1.set_ylabel('Average Accuracy (%)')
ax1.set_xticks(pmnist['tasks'])
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_ylim(0, 100)
ax1.text(0.02, 0.98, '(a) Permuted MNIST', transform=ax1.transAxes,
         fontsize=13, fontweight='bold', va='top')

# Right: Split CIFAR-100
for method, marker, color in [
    ('soft_csc', 'o', '#2196F3'), ('replay', 's', '#4CAF50'), ('packnet', '^', '#FF5722')]:
    label = {'soft_csc': 'Soft CSC', 'replay': 'Replay-only', 'packnet': 'PackNet'}[method]
    ax2.plot(cifar['tasks'], cifar[method], f'{marker}-', color=color,
             label=label, markersize=8, linewidth=2)

ax2.set_xlabel('Number of Tasks')
ax2.set_ylabel('Average Accuracy (%)')
ax2.set_xticks(cifar['tasks'])
ax2.legend()
ax2.grid(True, alpha=0.3)
ax2.set_ylim(60, 100)
ax2.text(0.02, 0.98, '(b) Split CIFAR-100', transform=ax2.transAxes,
         fontsize=13, fontweight='bold', va='top')

plt.tight_layout()
os.makedirs('figures', exist_ok=True)
plt.savefig('figures/scaling_comparison.pdf', bbox_inches='tight')
print("Saved figures/scaling_comparison.pdf")

# Also generate the LR control figure
fig2, ax = plt.subplots(figsize=(8, 5))

lr_data = {
    'lr=1e-3': 71.42, 'lr/3': 74.49, 'lr/9': 72.88, 'lr/15': 69.42,
}
soft_csc_val = 76.66  # best soft CSC at r=200

x = np.arange(len(lr_data))
bars = ax.bar(x, list(lr_data.values()), color='#4CAF50', alpha=0.7, label='Replay-only (tuned LR)')
ax.axhline(y=soft_csc_val, color='#2196F3', linewidth=2, linestyle='--',
           label=f'Soft CSC (g=0.01, b=1.0): {soft_csc_val}%')
ax.set_xticks(x)
ax.set_xticklabels(list(lr_data.keys()))
ax.set_ylabel('Average Accuracy (%)')
ax.set_xlabel('Learning Rate')
ax.legend()
ax.set_ylim(65, 80)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig('figures/lr_control.pdf', bbox_inches='tight')
print("Saved figures/lr_control.pdf")

# Pareto frontier figure
fig3, ax = plt.subplots(figsize=(8, 5))

pareto_data = [
    (100.0, 71.42, 'Replay-only r=200'),
    (24.5, 76.12, 'Soft CSC g=0.001'),
    (22.2, 76.77, 'Soft CSC g=0.005'),
    (19.1, 76.66, 'Soft CSC g=0.01'),
    (9.0, 76.10, 'Soft CSC g=0.01 (rel)'),
    (8.7, 69.67, 'CSC g=0.01 (no prot)'),
    (20.5, 71.00, 'CSC g=0.001 (no prot)'),
]

for bits, acc, label in pareto_data:
    if 'no prot' in label:
        ax.scatter(bits, acc, color='gray', s=60, zorder=5, alpha=0.6)
        ax.annotate(label, (bits, acc), fontsize=8, xytext=(5, -10),
                    textcoords='offset points', color='gray')
    elif 'Replay' in label:
        ax.scatter(bits, acc, color='#4CAF50', s=100, marker='s', zorder=5)
        ax.annotate(label, (bits, acc), fontsize=9, xytext=(5, 5),
                    textcoords='offset points')
    else:
        ax.scatter(bits, acc, color='#2196F3', s=80, zorder=5)
        ax.annotate(label, (bits, acc), fontsize=8, xytext=(5, -10),
                    textcoords='offset points')

# Connect Pareto-optimal points
pareto_pts = sorted([(19.1, 76.66), (22.2, 76.77), (24.5, 76.12)], key=lambda x: x[0])
ax.plot([p[0] for p in pareto_pts], [p[1] for p in pareto_pts], '--', color='#2196F3', alpha=0.5)

ax.set_xlabel('Bits Remaining (%)')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xlim(0, 110)
ax.set_ylim(65, 80)
ax.invert_xaxis()
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('figures/pareto_frontier.pdf', bbox_inches='tight')
print("Saved figures/pareto_frontier.pdf")
