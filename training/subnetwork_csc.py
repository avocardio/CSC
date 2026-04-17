"""Direction 1: Compression-driven sparse subnetwork discovery.

The quantization IS the mask. When b→0, the weight is effectively off.
Different tasks naturally discover different subnetworks through compression.

Key differences from previous CSC:
1. Higher gamma to create genuine sparsity (many b→0)
2. Save per-task bit-depth snapshots for evaluation
3. During eval of task k, restore task k's bit-depths (its subnetwork)
4. The overlap between task subnetworks = shared features

Direction 2: Compression-aware targeted replay.
Weight the replay loss per-sample based on which weights are at risk
of being overwritten (bit-depth changing most).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from models.resnet import QuantizedResNet18
from models.quantization import (
    CompressionGranularity, compute_average_bit_depth, get_compression_stats,
    get_quantizers, DifferentiableQuantizer,
)
from models.compression import apply_bias_l1_penalty
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from data.replay_buffer import ReplayBuffer
from data.der_buffer import DERBuffer


def save_bitdepth_snapshot(model):
    """Save current bit-depths as a task's subnetwork mask."""
    snapshot = {}
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer'):
            q = module.quantizer
            snapshot[name] = {
                'bit_depth': q.bit_depth.data.clone(),
                'exponent': q.exponent.data.clone(),
            }
    return snapshot


def restore_bitdepth_snapshot(model, snapshot):
    """Restore bit-depths from a saved snapshot (for task-specific eval)."""
    for name, module in model.named_modules():
        if name in snapshot and hasattr(module, 'quantizer'):
            q = module.quantizer
            q.bit_depth.data.copy_(snapshot[name]['bit_depth'])
            q.exponent.data.copy_(snapshot[name]['exponent'])


def evaluate_with_task_mask(model, test_loader, task_id, snapshot, device):
    """Evaluate using a specific task's bit-depth snapshot."""
    # Save current state
    current = save_bitdepth_snapshot(model)
    # Restore task-specific bit-depths
    restore_bitdepth_snapshot(model, snapshot)
    # Evaluate
    acc = evaluate_task(model, test_loader, task_id, device)
    # Restore current state
    restore_bitdepth_snapshot(model, current)
    return acc


def compute_weight_risk(model, prev_snapshot, current_snapshot):
    """Compute per-channel risk: how much each channel's bit-depth changed.

    High risk = the channel is being repurposed (bit-depth changing a lot).
    Used for Direction 2: targeted replay.
    """
    risks = {}
    for name in prev_snapshot:
        if name in current_snapshot:
            prev_b = prev_snapshot[name]['bit_depth']
            curr_b = current_snapshot[name]['bit_depth']
            # Risk = absolute change in bit-depth
            risks[name] = (curr_b - prev_b).abs()
    return risks


def train_subnetwork_csc(model, benchmark, config, device='cuda'):
    """Train with compression-driven subnetwork discovery.

    High gamma creates sparse subnetworks per task.
    Per-task bit-depth snapshots serve as continuous masks.
    """
    model = model.to(device)
    num_tasks = config.get('num_tasks', 10)
    epochs_per_task = config.get('epochs_per_task', 50)
    gamma = config.get('gamma', 0.05)  # Higher gamma for sparsity
    replay_per_task = config.get('replay_per_task', 200)
    replay_batch_size = config.get('replay_batch_size', 64)
    use_der = config.get('use_der', False)
    use_task_masks = config.get('use_task_masks', True)  # Eval with per-task masks
    use_targeted_replay = config.get('use_targeted_replay', False)  # Direction 2

    if use_der:
        buffer = DERBuffer(max_per_task=replay_per_task)
    else:
        buffer = ReplayBuffer(max_per_task=replay_per_task)

    cl_metrics = CLMetrics(num_tasks)
    task_snapshots = {}  # per-task bit-depth snapshots

    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_id} / {num_tasks-1}")
        print(f"{'='*60}")

        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Fresh optimizer per task
        quant_ids = set()
        qp, wp = [], []
        for m in model.modules():
            if isinstance(m, DifferentiableQuantizer):
                for p in m.parameters():
                    quant_ids.add(id(p)); qp.append(p)
        for p in model.parameters():
            if id(p) not in quant_ids:
                wp.append(p)

        optimizer = torch.optim.AdamW([
            {'params': wp, 'lr': 1e-3, 'weight_decay': 5e-4},
            {'params': qp, 'lr': 0.5, 'eps': 1e-3, 'weight_decay': 0},
        ])
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_task * len(train_loader))

        # Snapshot before training (for risk computation)
        pre_snapshot = save_bitdepth_snapshot(model)

        for epoch in range(epochs_per_task):
            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    task_loss = F.cross_entropy(logits, y)

                    # Compression loss — high gamma for sparsity
                    Q = compute_average_bit_depth(model)
                    comp_loss = gamma * Q
                    bias_loss = apply_bias_l1_penalty(model, 0.01)

                    loss = task_loss + comp_loss + bias_loss

                    # Replay
                    if buffer.size > 0:
                        if use_der:
                            buf = buffer.sample(replay_batch_size)
                            if buf is not None:
                                bx, by, blogits, bt = buf
                                bx, by, blogits, bt = bx.to(device), by.to(device), blogits.to(device), bt.to(device)

                                replay_loss = torch.tensor(0.0, device=device)
                                for tid in bt.unique():
                                    mask = bt == tid
                                    # If using task masks, restore that task's bit-depths for replay forward pass
                                    if use_task_masks and tid.item() in task_snapshots:
                                        current_bd = save_bitdepth_snapshot(model)
                                        restore_bitdepth_snapshot(model, task_snapshots[tid.item()])
                                        r_logits = model(bx[mask], task_id=tid.item())
                                        restore_bitdepth_snapshot(model, current_bd)
                                    else:
                                        r_logits = model(bx[mask], task_id=tid.item())

                                    replay_loss += F.cross_entropy(r_logits, by[mask])
                                    replay_loss += 0.5 * F.mse_loss(r_logits, blogits[mask])

                                loss = loss + replay_loss / len(bt.unique())
                        else:
                            buf = buffer.sample(replay_batch_size)
                            if buf is not None:
                                bx, by, bt = buf
                                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                                for tid in bt.unique():
                                    mask = bt == tid
                                    r_logits = model(bx[mask], task_id=tid.item())
                                    loss = loss + F.cross_entropy(r_logits, by[mask]) / len(bt.unique())

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Protect other task heads
                for pname, param in model.named_parameters():
                    if 'heads.' in pname and param.grad is not None:
                        head_idx = int(pname.split('heads.')[1].split('.')[0])
                        if head_idx != task_id:
                            param.grad.data.fill_(0)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            # End of epoch: recycle dead channels
            if (epoch + 1) % 10 == 0:
                recycled_count = 0
                with torch.no_grad():
                    for rname, rmodule in model.named_modules():
                        if hasattr(rmodule, 'quantizer') and hasattr(rmodule, 'conv'):
                            rq = rmodule.quantizer
                            rbits = rq.get_channel_bit_depths()
                            rdead = rbits < 0.5
                            if rdead.any():
                                n_d = rdead.sum().item()
                                nn.init.kaiming_normal_(
                                    rmodule.conv.weight.data[rdead],
                                    mode='fan_out', nonlinearity='relu')
                                alive_m = rbits[~rdead].mean() if (~rdead).any() else 4.0
                                rq.bit_depth.data[rdead] = alive_m
                                recycled_count += n_d
                if recycled_count > 0:
                    print(f"    Recycled {recycled_count} dead channels")

            if (epoch + 1) % 10 == 0 or epoch == 0:
                stats = get_compression_stats(model)
                acc = evaluate_task(model, test_loader, task_id, device)
                quantizers = get_quantizers(model)
                zero_channels = sum(
                    (q.get_channel_bit_depths() < 0.5).sum().item() for q in quantizers)
                total_channels = sum(q.num_output_channels for q in quantizers)

                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.4f} | "
                      f"Q: {stats['avg_bit_depth']:.2f} | Acc: {acc*100:.1f}% | "
                      f"Bits: {stats['compression_ratio']*100:.1f}% | "
                      f"Dead: {zero_channels}/{total_channels}")

        # Save task's bit-depth snapshot (its subnetwork)
        task_snapshots[task_id] = save_bitdepth_snapshot(model)

        # Report subnetwork overlap with previous tasks
        if task_id > 0:
            current_alive = set()
            for name, module in model.named_modules():
                if hasattr(module, 'quantizer'):
                    alive = (module.quantizer.get_channel_bit_depths() > 1.0)
                    for i in range(alive.shape[0]):
                        if alive[i]:
                            current_alive.add((name, i))

            for prev_t in range(task_id):
                prev_snap = task_snapshots[prev_t]
                prev_alive = set()
                for name in prev_snap:
                    bits = prev_snap[name]['bit_depth'].clamp(min=0)
                    for i in range(bits.shape[0]):
                        if bits[i] > 1.0:
                            prev_alive.add((name, i))

                overlap = len(current_alive & prev_alive)
                union = len(current_alive | prev_alive)
                iou = overlap / max(union, 1)
                print(f"  Subnetwork overlap task {task_id} & {prev_t}: "
                      f"IoU={iou:.3f} ({overlap} shared channels)")

        # Store replay
        if replay_per_task > 0:
            if use_der:
                task_data = benchmark.tasks[task_id]
                indices = task_data['train_indices']
                rng = np.random.RandomState(task_id)
                sel = rng.choice(len(indices), min(replay_per_task, len(indices)), replace=False)
                model.eval()
                samples = []
                with torch.no_grad():
                    for i in sel:
                        x_raw, y_raw = benchmark.train_dataset_raw[indices[i]]
                        local_y = task_data['class_mapping'][y_raw]
                        logits = model(x_raw.unsqueeze(0).to(device), task_id=task_id).squeeze(0).cpu()
                        samples.append((x_raw, local_y, logits, task_id))
                model.train()
                buffer.add_task_samples(samples)
            else:
                samples = benchmark.sample_for_replay(task_id, replay_per_task)
                buffer.add_task_samples(samples)

        # Evaluate all tasks
        print(f"\nEvaluating all tasks after Task {task_id}...")
        all_accs = []
        for t in range(task_id + 1):
            tl = benchmark.get_task_test_loader(t)
            if use_task_masks and t in task_snapshots:
                acc = evaluate_with_task_mask(model, tl, t, task_snapshots[t], device)
            else:
                acc = evaluate_task(model, tl, t, device)
            all_accs.append(acc)

        stats = get_compression_stats(model)
        cl_metrics.update(task_id, all_accs, stats['total_bits'])

        for j, acc in enumerate(all_accs):
            print(f"  Task {j}: {acc*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl_metrics.backward_transfer(task_id)*100:.2f}%")
        print(f"  Bits: {stats['compression_ratio']*100:.1f}%")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    cl_metrics.print_matrix()
    return cl_metrics, stats
