"""Post-hoc analysis of bit-depth trajectories for CSC paper figures.

Requires checkpoints with model_state_dict saved per task.
Analyzes: stability, feature reuse, consolidation scores.
All figures saved as publication-quality PDFs.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

plt.rcParams.update({
    'font.size': 12,
    'axes.labelsize': 14,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'legend.fontsize': 10,
    'figure.dpi': 150,
})


def load_bitdepths_from_checkpoints(ckpt_pattern, num_tasks, model_class, model_kwargs, device='cpu'):
    """Load bit-depth trajectories from per-task checkpoints.

    Returns dict: layer_name -> (num_tasks, num_channels) array of bit-depths.
    """
    trajectories = {}

    for t in range(num_tasks):
        path = ckpt_pattern.format(t)
        if not os.path.exists(path):
            print(f"  Missing checkpoint: {path}")
            continue

        ckpt = torch.load(path, map_location=device, weights_only=False)
        state = ckpt['model_state_dict']

        # Find quantizer bit-depth params
        for key in state:
            if 'quantizer.bit_depth' in key:
                layer_name = key.replace('.quantizer.bit_depth', '')
                bits = state[key].clamp(min=0).cpu().numpy()

                if layer_name not in trajectories:
                    trajectories[layer_name] = np.zeros((num_tasks, len(bits)))
                trajectories[layer_name][t] = bits

    return trajectories


def compute_stability_index(trajectories):
    """Compute stability index per channel: mean / std of bit-depth across tasks.

    Returns dict: layer_name -> (num_channels,) array of stability indices.
    """
    stability = {}
    for layer, traj in trajectories.items():
        means = traj.mean(axis=0)
        stds = traj.std(axis=0)
        # Avoid div by zero
        stds = np.maximum(stds, 1e-6)
        stability[layer] = means / stds
    return stability


def classify_channels(trajectories, high_threshold=6.0, low_threshold=2.0,
                      persistent_frac=0.8):
    """Classify channels as persistent, recycled, or decaying.

    Persistent: bit-depth > high_threshold for > persistent_frac of tasks
    Recycled: drops below low_threshold then rises above high_threshold-1
    Decaying: bit-depth monotonically decreases or stays low
    """
    categories = {}
    for layer, traj in trajectories.items():
        n_tasks, n_channels = traj.shape
        cats = {'persistent': [], 'recycled': [], 'decaying': [], 'other': []}

        for c in range(n_channels):
            ts = traj[:, c]
            high_frac = (ts > high_threshold).mean()
            ever_low = (ts < low_threshold).any()
            recovers = False
            if ever_low:
                low_idx = np.where(ts < low_threshold)[0][-1]
                if low_idx < n_tasks - 1:
                    recovers = (ts[low_idx+1:] > high_threshold - 1).any()

            if high_frac >= persistent_frac:
                cats['persistent'].append(c)
            elif ever_low and recovers:
                cats['recycled'].append(c)
            elif ts[-1] < low_threshold or (np.diff(ts) < 0).mean() > 0.7:
                cats['decaying'].append(c)
            else:
                cats['other'].append(c)

        categories[layer] = cats
    return categories


def plot_stability_by_layer(stability, trajectories, save_path='figures/stability_by_layer.pdf'):
    """Bar chart of mean stability index per layer."""
    layers = sorted(stability.keys())
    means = [stability[l].mean() for l in layers]

    # Shorten layer names
    short_names = []
    for l in layers:
        parts = l.split('.')
        short = '.'.join(p for p in parts if p not in ('conv', 'quantizer'))
        short_names.append(short)

    fig, ax = plt.subplots(figsize=(12, 5))
    x = np.arange(len(layers))
    ax.bar(x, means, color='steelblue', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(short_names, rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Stability Index (mean/std)')
    ax.set_xlabel('Layer')
    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_bitdepth_distributions(trajectories, task_indices, save_path='figures/bitdepth_distributions.pdf'):
    """Violin/histogram of bit-depth distributions at selected tasks."""
    layers = sorted(trajectories.keys())
    n_layers = len(layers)

    fig, axes = plt.subplots(1, len(task_indices), figsize=(5 * len(task_indices), 6), sharey=True)
    if len(task_indices) == 1:
        axes = [axes]

    for idx, (ax, t) in enumerate(zip(axes, task_indices)):
        all_bits = []
        layer_labels = []
        for l in layers:
            bits = trajectories[l][t]
            all_bits.append(bits)
            parts = l.split('.')
            layer_labels.append('.'.join(p for p in parts if p not in ('conv', 'quantizer')))

        parts = ax.violinplot(all_bits, positions=range(len(layers)), showmeans=True)
        ax.set_xticks(range(len(layers)))
        ax.set_xticklabels(layer_labels, rotation=45, ha='right', fontsize=8)
        ax.set_xlabel(f'Task {t}')
        if idx == 0:
            ax.set_ylabel('Bit Depth')

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_channel_trajectories(trajectories, categories, layer_name,
                              save_path='figures/channel_trajectories.pdf'):
    """Plot individual bit-depth trajectories for recycled channels."""
    if layer_name not in categories:
        print(f"  Layer {layer_name} not found")
        return

    cats = categories[layer_name]
    traj = trajectories[layer_name]
    n_tasks = traj.shape[0]

    fig, ax = plt.subplots(figsize=(12, 5))

    # Plot a few examples from each category
    colors = {'persistent': 'green', 'recycled': 'orange', 'decaying': 'red'}
    for cat_name, color in colors.items():
        indices = cats[cat_name][:5]  # up to 5 examples
        for c in indices:
            ax.plot(range(n_tasks), traj[:, c], color=color, alpha=0.6,
                    label=cat_name if c == indices[0] else None)

    ax.set_xlabel('Task')
    ax.set_ylabel('Bit Depth')
    ax.legend()
    ax.set_xlim(0, n_tasks - 1)
    ax.set_ylim(-0.5, 9)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_consolidation_scores(accuracy_matrices, method_names,
                              save_path='figures/consolidation_scores.pdf'):
    """Plot consolidation score (A[T][j] - A[j+5][j]) vs task number."""
    fig, ax = plt.subplots(figsize=(10, 5))

    for matrix, name in zip(accuracy_matrices, method_names):
        T = matrix.shape[0]
        scores = []
        tasks = []
        for j in range(T - 5):
            score = matrix[T - 1][j] - matrix[min(j + 5, T - 1)][j]
            scores.append(score * 100)
            tasks.append(j)
        ax.plot(tasks, scores, 'o-', label=name, markersize=3, alpha=0.7)

    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xlabel('Task')
    ax.set_ylabel('Consolidation Score (%)')
    ax.legend()

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


def plot_task_accuracy_over_time(accuracy_matrix, method_name, task_indices=None,
                                 save_path='figures/task_accuracy_over_time.pdf'):
    """Plot how each task's accuracy evolves as more tasks are trained."""
    T = accuracy_matrix.shape[0]
    if task_indices is None:
        # Plot every 5th task
        task_indices = list(range(0, T, max(1, T // 10)))

    fig, ax = plt.subplots(figsize=(12, 6))
    cmap = plt.cm.viridis(np.linspace(0, 1, len(task_indices)))

    for idx, (j, color) in enumerate(zip(task_indices, cmap)):
        traj = [accuracy_matrix[i][j] * 100 for i in range(j, T)]
        ax.plot(range(j, T), traj, color=color, alpha=0.7, linewidth=1.5,
                label=f'Task {j}')

    ax.set_xlabel('After Training on Task i')
    ax.set_ylabel('Accuracy (%)')
    ax.legend(fontsize=8, ncol=2)

    plt.tight_layout()
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"  Saved: {save_path}")


if __name__ == '__main__':
    print("Post-hoc analysis...")

    # Use available 10-task checkpoints for now
    from models.resnet import QuantizedResNet18
    from models.quantization import CompressionGranularity

    ckpt_pattern = 'checkpoints/hybrid_soft_task{}.pt'
    num_tasks = 10

    # Check how many checkpoints exist
    existing = [t for t in range(num_tasks) if os.path.exists(ckpt_pattern.format(t))]
    print(f"  Found {len(existing)} checkpoints for tasks {existing}")

    if len(existing) >= 5:
        print("\nLoading bit-depth trajectories...")
        traj = load_bitdepths_from_checkpoints(ckpt_pattern, num_tasks,
                                               QuantizedResNet18, {})

        if traj:
            print(f"  Loaded {len(traj)} layers")
            for l in sorted(traj.keys())[:3]:
                print(f"    {l}: shape={traj[l].shape}")

            print("\nComputing stability index...")
            stability = compute_stability_index(traj)
            plot_stability_by_layer(stability, traj)

            print("\nClassifying channels...")
            categories = classify_channels(traj)
            for l in sorted(categories.keys())[:5]:
                cats = categories[l]
                print(f"  {l}: persistent={len(cats['persistent'])}, "
                      f"recycled={len(cats['recycled'])}, "
                      f"decaying={len(cats['decaying'])}, "
                      f"other={len(cats['other'])}")

            print("\nPlotting distributions...")
            task_pts = [0, num_tasks // 2, num_tasks - 1]
            plot_bitdepth_distributions(traj, task_pts)

            # Find a layer with recycled channels
            for l in sorted(categories.keys()):
                if len(categories[l]['recycled']) > 0:
                    print(f"\nPlotting trajectories for {l} ({len(categories[l]['recycled'])} recycled)...")
                    plot_channel_trajectories(traj, categories, l)
                    break
        else:
            print("  No quantizer parameters found in checkpoints")

    # Accuracy analysis from 50-task runs
    for fname, name in [
        ('checkpoints/pmnist_soft_csc_t50.pt', 'Soft CSC 50-task'),
        ('checkpoints/pmnist_replay_t50.pt', 'Replay 50-task'),
    ]:
        if os.path.exists(fname):
            d = torch.load(fname, map_location='cpu', weights_only=False)
            if 'accuracy_matrix' in d:
                plot_task_accuracy_over_time(d['accuracy_matrix'], name,
                    save_path=f'figures/task_acc_over_time_{name.replace(" ", "_")}.pdf')

    print("\nDone!")
