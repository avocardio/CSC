"""Clean Pareto frontier figure for the paper."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import os

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 9, 'figure.dpi': 150,
    'font.family': 'serif',
})

os.makedirs('figures', exist_ok=True)

fig, ax = plt.subplots(figsize=(8, 5.5))

# 10-task Split CIFAR-100 results
# Format: (bits%, acc%, label, color, marker, size)
points = [
    # Baselines (100% bits)
    (100, 71.4, 'Replay r=200', '#4CAF50', 's', 80),
    (100, 77.9, 'DER++ r=200', '#9C27B0', 'D', 80),
    (100, 56.9, 'EWC', '#795548', 'v', 60),
    (100, 64.5, 'SI', '#607D8B', 'v', 60),

    # CSC without soft protection
    (20.5, 71.0, 'CSC (no prot)', '#90CAF9', 'o', 60),
    (8.7, 69.7, 'CSC g=0.01', '#BBDEFB', 'o', 50),

    # Soft CSC (our method)
    (24.5, 76.1, 'Soft CSC r=200', '#2196F3', 'o', 100),
    (24.5, 78.4, 'Soft CSC+DER++ r=200', '#1565C0', 'o', 100),
    (24.5, 83.0, 'Soft CSC+DER++ r=500', '#0D47A1', 'o', 120),

    # PackNet
    (94.4, 87.2, 'PackNet', '#FF5722', '^', 120),
]

for bits, acc, label, color, marker, size in points:
    ax.scatter(bits, acc, color=color, s=size, marker=marker, zorder=5,
               edgecolors='white', linewidths=0.5)

# Annotations with smart positioning
annotations = {
    'Replay r=200': (5, -12),
    'DER++ r=200': (5, 5),
    'EWC': (-35, -12),
    'SI': (-20, 8),
    'CSC (no prot)': (-50, -12),
    'CSC g=0.01': (5, -10),
    'Soft CSC r=200': (-70, 5),
    'Soft CSC+DER++ r=200': (-90, -10),
    'Soft CSC+DER++ r=500': (-95, 5),
    'PackNet': (-45, 8),
}

for bits, acc, label, color, marker, size in points:
    offset = annotations.get(label, (5, 5))
    ax.annotate(label, (bits, acc), fontsize=7.5, xytext=offset,
                textcoords='offset points', color='#333333',
                arrowprops=dict(arrowstyle='-', color='#CCCCCC', lw=0.5) if abs(offset[0]) > 30 else None)

# Highlight the Pareto frontier
pareto_pts = [(8.7, 69.7), (20.5, 71.0), (24.5, 76.1), (24.5, 78.4), (24.5, 83.0), (94.4, 87.2)]
# Draw shaded region showing CSC advantage
ax.fill_between([0, 30], [60, 60], [90, 90], alpha=0.05, color='#2196F3')
ax.fill_between([85, 105], [60, 60], [90, 90], alpha=0.05, color='#FF5722')

ax.annotate('CSC region\n(compact)', xy=(15, 62), fontsize=9, color='#2196F3',
            ha='center', style='italic', alpha=0.7)
ax.annotate('Full-size\nmethods', xy=(97, 62), fontsize=9, color='#FF5722',
            ha='center', style='italic', alpha=0.7)

ax.set_xlabel('Model Capacity (% of original bits)')
ax.set_ylabel('Average Accuracy (%)')
ax.set_xlim(-5, 110)
ax.set_ylim(52, 92)
ax.grid(True, alpha=0.2)

plt.tight_layout()
plt.savefig('figures/pareto_frontier.pdf', bbox_inches='tight')
print("Saved figures/pareto_frontier.pdf")
