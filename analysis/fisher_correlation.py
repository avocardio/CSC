"""Analyze correlation between learned bit-depths and Fisher information.

If bit-depths correlate with Fisher, it validates that self-compression
discovers parameter importance as a free byproduct — the same signal that
EWC computes explicitly at significant cost.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.dpi': 150,
    'font.family': 'serif',
})


def compute_channel_fisher(model, data_loader, task_id, device, n_samples=1000):
    """Compute diagonal Fisher information per channel.

    Returns dict: layer_name -> (num_channels,) Fisher values.
    """
    model.eval()
    fisher = {}

    # Initialize
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer'):
            n_channels = module.quantizer.num_output_channels
            fisher[name] = torch.zeros(n_channels, device=device)

    count = 0
    for batch in data_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        logits = model(x, task_id=task_id)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        # Accumulate squared gradients per channel
        for name, module in model.named_modules():
            if name in fisher and hasattr(module, 'conv'):
                if module.conv.weight.grad is not None:
                    # Average squared gradient per output channel
                    grad_sq = module.conv.weight.grad.data ** 2
                    channel_fisher = grad_sq.view(grad_sq.shape[0], -1).mean(dim=1)
                    fisher[name] += channel_fisher

        count += x.size(0)
        if count >= n_samples:
            break

    for name in fisher:
        fisher[name] /= count

    model.train()
    return fisher


def get_channel_bitdepths(model):
    """Extract current channel bit-depths from model."""
    bitdepths = {}
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer'):
            bitdepths[name] = module.quantizer.get_channel_bit_depths().detach().cpu().numpy()
    return bitdepths


def analyze_fisher_bitdepth_correlation(model, benchmark, task_id, device='cuda'):
    """Compute and correlate Fisher information with bit-depths after a task."""
    from training.metrics import evaluate_task

    train_loader = benchmark.get_task_dataloaders(task_id)[0]

    # Get Fisher
    fisher = compute_channel_fisher(model, train_loader, task_id, device)

    # Get bit-depths
    bitdepths = get_channel_bitdepths(model)

    # Compute correlation per layer and overall
    all_fisher = []
    all_bitdepths = []
    layer_corrs = {}

    for name in fisher:
        if name in bitdepths:
            f = fisher[name].cpu().numpy()
            b = bitdepths[name]
            if len(f) == len(b) and len(f) > 1:
                corr, pval = stats.spearmanr(f, b)
                layer_corrs[name] = (corr, pval)
                all_fisher.extend(f.tolist())
                all_bitdepths.extend(b.tolist())

    # Overall correlation
    if len(all_fisher) > 1:
        overall_corr, overall_pval = stats.spearmanr(all_fisher, all_bitdepths)
    else:
        overall_corr, overall_pval = 0, 1

    return {
        'layer_corrs': layer_corrs,
        'overall_corr': overall_corr,
        'overall_pval': overall_pval,
        'all_fisher': np.array(all_fisher),
        'all_bitdepths': np.array(all_bitdepths),
    }


def run_fisher_analysis(num_tasks=10, epochs_per_task=50, device='cuda'):
    """Run Fisher-bitdepth correlation analysis across tasks."""
    from models.resnet import QuantizedResNet18
    from models.quantization import CompressionGranularity, compute_average_bit_depth, DifferentiableQuantizer
    from data.split_cifar100 import SplitCIFAR100
    from data.replay_buffer import ReplayBuffer
    from training.hybrid import SoftProtectionCSC, make_optimizer_adamw
    from models.compression import apply_bias_l1_penalty

    torch.manual_seed(42); torch.cuda.manual_seed(42)
    benchmark = SplitCIFAR100(num_tasks=num_tasks, batch_size=128, num_workers=8, seed=42)
    classes_per_task = 100 // num_tasks
    model = QuantizedResNet18(classes_per_task, num_tasks,
                              granularity=CompressionGranularity.CHANNEL).to(device)
    protector = SoftProtectionCSC(model, beta=1.0, device=device)
    rb = ReplayBuffer(max_per_task=200)

    correlations_over_tasks = []

    for task_id in range(num_tasks):
        print(f"\nTask {task_id}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        optimizer = make_optimizer_adamw(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_task * len(train_loader))
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(epochs_per_task):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)
                    Q = compute_average_bit_depth(model)
                    loss = loss + 0.001 * Q + apply_bias_l1_penalty(model, 0.01)
                    if rb.size > 0:
                        rd = rb.sample(64)
                        if rd:
                            rx, ry, rt = rd
                            rx, ry, rt = rx.to(device), ry.to(device), rt.to(device)
                            rl = torch.tensor(0.0, device=device)
                            for tid in rt.unique():
                                m = rt == tid
                                rl += F.cross_entropy(model(rx[m], task_id=tid.item()), ry[m])
                            loss = loss + rl / len(rt.unique())
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                protector.scale_gradients(task_id)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        rb.add_task_samples(benchmark.sample_for_replay(task_id, 200))

        # Analyze correlation
        result = analyze_fisher_bitdepth_correlation(model, benchmark, task_id, device)
        correlations_over_tasks.append(result)
        print(f"  Fisher-BitDepth correlation: r={result['overall_corr']:.4f} "
              f"(p={result['overall_pval']:.4e})")

    # Plot correlation over tasks
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Left: correlation over tasks
    corrs = [r['overall_corr'] for r in correlations_over_tasks]
    ax1.plot(range(num_tasks), corrs, 'o-', color='#2196F3', markersize=6)
    ax1.set_xlabel('Task')
    ax1.set_ylabel('Spearman Correlation (Fisher vs Bit-Depth)')
    ax1.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(-0.3, 1.0)

    # Right: scatter plot for final task
    final = correlations_over_tasks[-1]
    ax2.scatter(np.log10(final['all_fisher'] + 1e-10), final['all_bitdepths'],
                alpha=0.3, s=10, color='#2196F3')
    ax2.set_xlabel('log₁₀(Fisher Information)')
    ax2.set_ylabel('Learned Bit-Depth')
    ax2.text(0.05, 0.95, f'ρ = {final["overall_corr"]:.3f}\np = {final["overall_pval"]:.2e}',
             transform=ax2.transAxes, fontsize=11, va='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('figures/fisher_correlation.pdf', bbox_inches='tight')
    plt.close()
    print("\nSaved figures/fisher_correlation.pdf")

    return correlations_over_tasks


if __name__ == '__main__':
    print("Running Fisher-BitDepth correlation analysis...")
    results = run_fisher_analysis(num_tasks=10, epochs_per_task=50)
