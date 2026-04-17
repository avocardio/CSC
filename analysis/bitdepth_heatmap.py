"""Bit-depth evolution heatmap visualization.

Signature figure: shows how bit depths per channel per layer
evolve across training and across tasks.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from models.quantization import get_quantizers


def collect_bitdepth_snapshot(model):
    """Collect current bit depths from all layers."""
    snapshot = {}
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer'):
            q = module.quantizer
            channel_bits = q.get_channel_bit_depths().detach().cpu().numpy()
            snapshot[name] = channel_bits
    return snapshot


class BitDepthTracker:
    """Track bit depth evolution over training for visualization."""

    def __init__(self):
        self.snapshots = []  # list of (step, task_id, snapshot_dict)

    def record(self, step, task_id, model):
        snapshot = collect_bitdepth_snapshot(model)
        self.snapshots.append((step, task_id, snapshot))

    def plot_heatmap(self, save_path=None, layer_names=None):
        """Plot bit-depth evolution heatmap.

        X-axis: training steps/epochs
        Y-axis: channels (grouped by layer)
        Color: bit depth
        Vertical lines: task boundaries
        """
        if not self.snapshots:
            print("No snapshots recorded")
            return

        # Get layer names from first snapshot
        _, _, first_snap = self.snapshots[0]
        if layer_names is None:
            layer_names = list(first_snap.keys())

        # Build the data matrix
        n_steps = len(self.snapshots)
        total_channels = sum(len(first_snap[ln]) for ln in layer_names if ln in first_snap)

        data = np.zeros((total_channels, n_steps))
        steps = []
        task_boundaries = []

        prev_task = -1
        for col, (step, task_id, snap) in enumerate(self.snapshots):
            steps.append(step)
            if task_id != prev_task:
                task_boundaries.append(col)
                prev_task = task_id

            row = 0
            for ln in layer_names:
                if ln in snap:
                    bits = snap[ln]
                    data[row:row + len(bits), col] = bits
                    row += len(bits)

        fig, ax = plt.subplots(figsize=(14, 8))
        im = ax.imshow(data, aspect='auto', cmap='YlOrRd_r',
                       interpolation='nearest', vmin=0, vmax=8)

        # Task boundary lines
        for tb in task_boundaries:
            ax.axvline(x=tb, color='blue', linewidth=2, linestyle='--', alpha=0.7)

        # Layer boundary lines
        row = 0
        for ln in layer_names:
            if ln in first_snap:
                row += len(first_snap[ln])
                ax.axhline(y=row - 0.5, color='white', linewidth=0.5, alpha=0.5)

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Channel Index (grouped by layer)')
        ax.set_title('Bit Depth Evolution Across Training')

        # Set x ticks to show step numbers
        n_ticks = min(10, n_steps)
        tick_indices = np.linspace(0, n_steps - 1, n_ticks, dtype=int)
        ax.set_xticks(tick_indices)
        ax.set_xticklabels([str(steps[i]) for i in tick_indices], rotation=45)

        plt.colorbar(im, ax=ax, label='Bit Depth')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved heatmap to {save_path}")
        plt.close()

    def plot_layer_summary(self, save_path=None):
        """Plot average bit depth per layer over training."""
        if not self.snapshots:
            return

        _, _, first_snap = self.snapshots[0]
        layer_names = list(first_snap.keys())

        fig, ax = plt.subplots(figsize=(12, 6))
        steps = [s[0] for s in self.snapshots]

        for ln in layer_names:
            avg_bits = [snap[2].get(ln, np.array([0])).mean() for snap in self.snapshots]
            short_name = ln.split('.')[-1] if '.' in ln else ln
            ax.plot(steps, avg_bits, label=short_name, alpha=0.7)

        # Task boundaries
        prev_task = -1
        for step, task_id, _ in self.snapshots:
            if task_id != prev_task:
                ax.axvline(x=step, color='gray', linestyle='--', alpha=0.5)
                prev_task = task_id

        ax.set_xlabel('Training Step')
        ax.set_ylabel('Average Bit Depth')
        ax.set_title('Average Bit Depth per Layer')
        ax.legend(fontsize=6, ncol=3)
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved layer summary to {save_path}")
        plt.close()
