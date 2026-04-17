"""Wow-factor analyses for the CSC paper.

1. Forward transfer: does compression improve zero-shot performance on unseen tasks?
2. Forgetting prediction: can bit-depth dynamics predict which tasks will be forgotten?
3. Task similarity from bit-depths: extract task affinity matrix for free.
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

plt.rcParams.update({
    'font.size': 12, 'axes.labelsize': 14, 'xtick.labelsize': 11,
    'ytick.labelsize': 11, 'legend.fontsize': 10, 'figure.dpi': 150,
    'font.family': 'serif',
})

os.makedirs('figures', exist_ok=True)


def measure_forward_transfer(model_class, model_kwargs, benchmark, trainer_fn,
                             config, num_tasks, device='cuda'):
    """Measure forward transfer: accuracy on task t BEFORE training on it.

    Returns:
        fwd_transfer: list of accuracies on each task before seeing it
        final_accs: list of accuracies on each task after training on it
    """
    from training.metrics import evaluate_task

    model = model_class(**model_kwargs).to(device)
    fwd_transfer = []
    final_accs = []

    # Can't measure FT for task 0 (no prior training)
    fwd_transfer.append(None)

    trainer_state = trainer_fn(model, benchmark, config, device, return_per_task=True)
    return trainer_state


def plot_forward_transfer_comparison(ft_csc, ft_replay, ft_packnet, save_path='figures/forward_transfer.pdf'):
    """Plot forward transfer comparison across methods."""
    fig, ax = plt.subplots(figsize=(8, 5))

    methods = [
        ('Soft CSC', ft_csc, '#2196F3', 'o'),
        ('Replay-only', ft_replay, '#4CAF50', 's'),
    ]
    if ft_packnet is not None:
        methods.append(('PackNet', ft_packnet, '#FF5722', '^'))

    for name, ft, color, marker in methods:
        tasks = range(1, len(ft))
        vals = [ft[t] * 100 for t in tasks if ft[t] is not None]
        ax.plot(range(1, len(vals) + 1), vals, f'{marker}-', color=color,
                label=name, markersize=6, linewidth=1.5)

    ax.set_xlabel('Task Number')
    ax.set_ylabel('Zero-Shot Accuracy Before Training (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def analyze_forgetting_prediction(accuracy_matrix, bitdepth_trajectories, num_tasks):
    """Analyze whether bit-depth changes predict forgetting.

    For each task j, compute:
    - The mean bit-depth change of channels that were important for task j
      during training of task j+1
    - The accuracy drop on task j after training task j+1

    If these correlate, bit-depth dynamics predict forgetting.
    """
    forgetting = []
    bitdepth_change = []

    for j in range(num_tasks - 1):
        # Forgetting: A[j][j] - A[j+1][j]
        forg = accuracy_matrix[j][j] - accuracy_matrix[j + 1][j]
        forgetting.append(forg)

        # Bit-depth change during task j+1 for channels important to task j
        if bitdepth_trajectories is not None:
            # Average absolute change in bit-depth from task j to task j+1
            changes = []
            for layer, traj in bitdepth_trajectories.items():
                if j + 1 < traj.shape[0]:
                    delta = np.abs(traj[j + 1] - traj[j]).mean()
                    changes.append(delta)
            bitdepth_change.append(np.mean(changes) if changes else 0)

    return forgetting, bitdepth_change


def plot_forgetting_prediction(forgetting, bitdepth_change,
                                save_path='figures/forgetting_prediction.pdf'):
    """Scatter plot: bit-depth change vs forgetting."""
    fig, ax = plt.subplots(figsize=(6, 5))

    ax.scatter(bitdepth_change, [f * 100 for f in forgetting],
               color='#2196F3', s=60, alpha=0.7, zorder=5)

    # Fit line
    if len(bitdepth_change) > 2:
        z = np.polyfit(bitdepth_change, [f * 100 for f in forgetting], 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(bitdepth_change), max(bitdepth_change), 100)
        ax.plot(x_line, p(x_line), '--', color='red', alpha=0.5)

        # Correlation
        corr = np.corrcoef(bitdepth_change, forgetting)[0, 1]
        ax.text(0.05, 0.95, f'r = {corr:.3f}', transform=ax.transAxes,
                fontsize=12, va='top')

    ax.set_xlabel('Mean Bit-Depth Change')
    ax.set_ylabel('Forgetting (%)')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def compute_task_similarity_from_bitdepths(bitdepth_trajectories, num_tasks):
    """Compute task similarity matrix from shared high-bit-depth channels.

    For tasks i and j, similarity = cosine similarity of their bit-depth vectors.
    """
    # Concatenate all layers into a single vector per task
    task_vectors = []
    for t in range(num_tasks):
        vec = []
        for layer in sorted(bitdepth_trajectories.keys()):
            traj = bitdepth_trajectories[layer]
            if t < traj.shape[0]:
                vec.append(traj[t])
        task_vectors.append(np.concatenate(vec))

    task_vectors = np.array(task_vectors)

    # Cosine similarity matrix
    from sklearn.metrics.pairwise import cosine_similarity
    sim_matrix = cosine_similarity(task_vectors)

    return sim_matrix


def plot_task_similarity(sim_matrix, save_path='figures/task_similarity.pdf'):
    """Plot task similarity heatmap."""
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(sim_matrix, cmap='RdYlBu_r', vmin=0.9, vmax=1.0)

    n = sim_matrix.shape[0]
    ax.set_xticks(range(0, n, max(1, n // 10)))
    ax.set_yticks(range(0, n, max(1, n // 10)))
    ax.set_xlabel('Task')
    ax.set_ylabel('Task')
    plt.colorbar(im, label='Cosine Similarity')
    plt.tight_layout()
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Saved {save_path}")


def run_forward_transfer_experiment(num_tasks=20, epochs_per_task=30, device='cuda'):
    """Run forward transfer measurement for Soft CSC and Replay-only."""
    from models.resnet import QuantizedResNet18
    from models.quantization import CompressionGranularity, compute_average_bit_depth, DifferentiableQuantizer
    from data.split_cifar100 import SplitCIFAR100
    from data.replay_buffer import ReplayBuffer
    from training.metrics import evaluate_task
    from training.hybrid import SoftProtectionCSC, make_optimizer_adamw
    from models.compression import apply_bias_l1_penalty
    from baselines.finetune import SimpleResNet18

    benchmark = SplitCIFAR100(num_tasks=num_tasks, batch_size=128, num_workers=8, seed=42)
    classes_per_task = 100 // num_tasks

    results = {}

    # ---- Soft CSC ----
    print("\n=== Soft CSC Forward Transfer ===")
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    model = QuantizedResNet18(classes_per_task, num_tasks,
                              granularity=CompressionGranularity.CHANNEL).to(device)
    protector = SoftProtectionCSC(model, beta=1.0, device=device)
    rb = ReplayBuffer(max_per_task=200)

    csc_ft = [None]  # no FT for task 0
    csc_final = []

    for task_id in range(num_tasks):
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Measure forward transfer (before training on this task)
        if task_id > 0:
            ft_acc = evaluate_task(model, test_loader, task_id, device)
            csc_ft.append(ft_acc)
            print(f"  Task {task_id} FT (before training): {ft_acc*100:.1f}%")

        # Train
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

        final_acc = evaluate_task(model, test_loader, task_id, device)
        csc_final.append(final_acc)
        rb.add_task_samples(benchmark.sample_for_replay(task_id, 200))

        if (task_id + 1) % 5 == 0:
            print(f"  Task {task_id} final: {final_acc*100:.1f}%")

    results['soft_csc'] = {'ft': csc_ft, 'final': csc_final}

    # ---- Replay-only ----
    print("\n=== Replay-only Forward Transfer ===")
    torch.manual_seed(42); torch.cuda.manual_seed(42)
    model_r = SimpleResNet18(classes_per_task, num_tasks).to(device)
    rb_r = ReplayBuffer(max_per_task=200)

    replay_ft = [None]
    replay_final = []

    for task_id in range(num_tasks):
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        if task_id > 0:
            ft_acc = evaluate_task(model_r, test_loader, task_id, device)
            replay_ft.append(ft_acc)

        optimizer = torch.optim.Adam(model_r.parameters(), lr=1e-3, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_task * len(train_loader))
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(epochs_per_task):
            model_r.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model_r(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)
                    if rb_r.size > 0:
                        rd = rb_r.sample(64)
                        if rd:
                            rx, ry, rt = rd
                            rx, ry, rt = rx.to(device), ry.to(device), rt.to(device)
                            rl = torch.tensor(0.0, device=device)
                            for tid in rt.unique():
                                m = rt == tid
                                rl += F.cross_entropy(model_r(rx[m], task_id=tid.item()), ry[m])
                            loss = loss + rl / len(rt.unique())
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

        final_acc = evaluate_task(model_r, test_loader, task_id, device)
        replay_final.append(final_acc)
        rb_r.add_task_samples(benchmark.sample_for_replay(task_id, 200))

        if (task_id + 1) % 5 == 0:
            print(f"  Task {task_id} final: {final_acc*100:.1f}%")

    results['replay'] = {'ft': replay_ft, 'final': replay_final}

    # Compute forward transfer metrics
    for method in ['soft_csc', 'replay']:
        ft_vals = [v for v in results[method]['ft'] if v is not None]
        random_baseline = 1.0 / classes_per_task
        mean_ft = np.mean(ft_vals) - random_baseline
        print(f"\n{method} Forward Transfer: mean={mean_ft*100:.2f}% above random ({random_baseline*100:.0f}%)")
        print(f"  Per-task: {[f'{v*100:.1f}' for v in ft_vals]}")

    # Plot
    fig, ax = plt.subplots(figsize=(9, 5))
    for method, color, marker, label in [
        ('soft_csc', '#2196F3', 'o', 'Soft CSC'),
        ('replay', '#4CAF50', 's', 'Replay-only'),
    ]:
        ft = results[method]['ft']
        vals = [ft[t] * 100 for t in range(1, len(ft)) if ft[t] is not None]
        ax.plot(range(1, len(vals) + 1), vals, f'{marker}-', color=color,
                label=label, markersize=5, linewidth=1.5)

    ax.axhline(y=100.0 / classes_per_task, color='gray', linestyle='--', alpha=0.5,
               label=f'Random ({100//classes_per_task}%)')
    ax.set_xlabel('Task Number')
    ax.set_ylabel('Zero-Shot Accuracy Before Training (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('figures/forward_transfer.pdf', bbox_inches='tight')
    plt.close()
    print("Saved figures/forward_transfer.pdf")

    return results


if __name__ == '__main__':
    print("Running wow-factor analyses...")
    results = run_forward_transfer_experiment(num_tasks=20, epochs_per_task=30)
