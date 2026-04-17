"""Capacity utilization and accuracy curves for CL experiments."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib.pyplot as plt


def plot_accuracy_over_tasks(results_dict, save_path=None):
    """Plot average accuracy after each task for multiple methods.

    Args:
        results_dict: {method_name: CLMetrics or accuracy_matrix}
        save_path: path to save figure
    """
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in results_dict.items():
        if hasattr(data, 'accuracy_matrix'):
            matrix = data.accuracy_matrix
        else:
            matrix = data

        num_tasks = matrix.shape[0]
        avg_accs = []
        for t in range(num_tasks):
            avg_accs.append(np.mean(matrix[t, :t + 1]) * 100)

        ax.plot(range(1, num_tasks + 1), avg_accs, 'o-', label=name, markersize=6)

    ax.set_xlabel('Number of Tasks Learned')
    ax.set_ylabel('Average Accuracy (%)')
    ax.set_title('Average Accuracy vs Number of Tasks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_capacity_utilization(bits_histories, method_names, save_path=None):
    """Plot bits used over tasks for different methods."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for bits, name in zip(bits_histories, method_names):
        tasks = range(1, len(bits) + 1)
        ax.plot(tasks, bits, 'o-', label=name, markersize=6)

    ax.set_xlabel('Number of Tasks Learned')
    ax.set_ylabel('Total Bits Used')
    ax.set_title('Capacity Utilization Over Tasks')
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_backward_transfer(results_dict, save_path=None):
    """Plot backward transfer after each task."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for name, data in results_dict.items():
        if hasattr(data, 'accuracy_matrix'):
            matrix = data.accuracy_matrix
        else:
            matrix = data

        num_tasks = matrix.shape[0]
        bwts = [0]
        for t in range(1, num_tasks):
            bwt = np.mean([matrix[t, j] - matrix[j, j] for j in range(t)]) * 100
            bwts.append(bwt)

        ax.plot(range(1, num_tasks + 1), bwts, 'o-', label=name, markersize=6)

    ax.set_xlabel('Number of Tasks Learned')
    ax.set_ylabel('Backward Transfer (%)')
    ax.set_title('Backward Transfer Over Tasks')
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.legend()
    ax.grid(True, alpha=0.3)

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def plot_forgetting_matrix(accuracy_matrix, save_path=None, title='Accuracy Matrix'):
    """Plot the full accuracy matrix as a heatmap."""
    num_tasks = accuracy_matrix.shape[0]
    fig, ax = plt.subplots(figsize=(8, 7))

    # Mask upper triangle (not yet trained)
    mask = np.triu(np.ones_like(accuracy_matrix, dtype=bool), k=1)
    data = np.ma.array(accuracy_matrix * 100, mask=mask)

    im = ax.imshow(data, cmap='RdYlGn', vmin=0, vmax=100, interpolation='nearest')

    for i in range(num_tasks):
        for j in range(num_tasks):
            if not mask[i, j]:
                ax.text(j, i, f'{accuracy_matrix[i, j]*100:.1f}',
                       ha='center', va='center', fontsize=8)

    ax.set_xlabel('Task j (evaluated on)')
    ax.set_ylabel('After training on Task i')
    ax.set_title(title)
    ax.set_xticks(range(num_tasks))
    ax.set_yticks(range(num_tasks))
    plt.colorbar(im, label='Accuracy (%)')
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
