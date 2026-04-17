"""Direction 2: Compression-aware targeted protection.

Use bit-depth dynamics to detect at-risk knowledge and apply stronger
protection to weights whose bit-depth is dropping (being overwritten).

The key insight: when a weight's bit-depth decreases during new task training,
it means the compression objective has decided that weight is less important
for the current task — but it may have been important for previous tasks.
This is a PREDICTION of imminent forgetting.

We use this signal to:
1. Apply stronger gradient scaling to at-risk weights (higher effective protection)
2. The protection is DYNAMIC — it changes as the compression landscape evolves
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
import numpy as np
from models.quantization import (
    compute_average_bit_depth, get_compression_stats, get_quantizers,
    DifferentiableQuantizer,
)
from models.compression import apply_bias_l1_penalty
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.replay_buffer import ReplayBuffer
from data.der_buffer import DERBuffer


class DynamicProtection:
    """Dynamically adjust protection based on bit-depth change rate.

    At each step, compare current bit-depths to a reference snapshot.
    Weights whose bit-depth is DECREASING are at risk of losing old knowledge.
    Apply extra protection (lower effective LR) to at-risk weights.

    Protection formula:
    scale = 1 / (1 + beta_base * b_i + beta_risk * max(0, b_ref - b_i))

    Where b_ref is the bit-depth at the start of the current task.
    The second term adds extra protection proportional to how much the
    bit-depth has decreased (risk of forgetting).
    """

    def __init__(self, model, beta_base=0.5, beta_risk=2.0, device='cuda'):
        self.model = model
        self.beta_base = beta_base
        self.beta_risk = beta_risk
        self.device = device
        self.reference_bitdepths = {}  # snapshot at task start

    def snapshot_reference(self):
        """Save current bit-depths as reference for risk computation."""
        self.reference_bitdepths = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer'):
                self.reference_bitdepths[name] = module.quantizer.get_channel_bit_depths().detach().clone()

    def scale_gradients(self, task_id):
        """Apply dynamic protection based on bit-depth change."""
        for name, module in self.model.named_modules():
            if not hasattr(module, 'quantizer'):
                continue

            q = module.quantizer
            current_bits = q.get_channel_bit_depths().detach()

            # Base protection from current bit-depth
            base_scale = self.beta_base * current_bits

            # Risk-based protection from bit-depth decrease
            if name in self.reference_bitdepths:
                ref_bits = self.reference_bitdepths[name]
                # How much has bit-depth decreased? (positive = at risk)
                decrease = (ref_bits - current_bits).clamp(min=0)
                risk_scale = self.beta_risk * decrease
            else:
                risk_scale = torch.zeros_like(current_bits)

            total_scale = 1.0 / (1.0 + base_scale + risk_scale)

            # Apply to conv weight gradients
            if hasattr(module, 'conv') and module.conv.weight.grad is not None:
                weight = module.conv.weight
                if weight.dim() == 4:
                    s = total_scale.view(-1, 1, 1, 1)
                elif weight.dim() == 2:
                    s = total_scale.view(-1, 1)
                else:
                    continue
                weight.grad.data *= s.expand_as(weight.grad)

            # Scale bit-depth gradients
            if q.bit_depth.grad is not None:
                q.bit_depth.grad.data *= total_scale
            if q.exponent.grad is not None:
                q.exponent.grad.data *= total_scale

        # Protect other task heads
        for pname, param in self.model.named_parameters():
            if 'heads.' in pname and param.grad is not None:
                head_idx = int(pname.split('heads.')[1].split('.')[0])
                if head_idx != task_id:
                    param.grad.data.fill_(0)


def train_targeted_csc(model, benchmark, config, device='cuda'):
    """Train with dynamic compression-aware protection."""
    model = model.to(device)
    num_tasks = config.get('num_tasks', 10)
    epochs_per_task = config.get('epochs_per_task', 50)
    gamma = config.get('gamma', 0.001)
    beta_base = config.get('beta_base', 0.5)
    beta_risk = config.get('beta_risk', 2.0)
    replay_per_task = config.get('replay_per_task', 200)
    use_der = config.get('use_der', False)

    if use_der:
        buffer = DERBuffer(max_per_task=replay_per_task)
    else:
        buffer = ReplayBuffer(max_per_task=replay_per_task)

    cl_metrics = CLMetrics(num_tasks)
    protector = DynamicProtection(model, beta_base=beta_base,
                                  beta_risk=beta_risk, device=device)

    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_id} / {num_tasks-1}")
        print(f"{'='*60}")

        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Snapshot bit-depths at task start (reference for risk)
        protector.snapshot_reference()

        # Fresh optimizer
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

        for epoch in range(epochs_per_task):
            model.train()
            epoch_loss = 0
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                    Q = compute_average_bit_depth(model)
                    loss = loss + gamma * Q + apply_bias_l1_penalty(model, 0.01)

                    # Replay
                    if buffer.size > 0:
                        if use_der:
                            buf = buffer.sample(64)
                            if buf is not None:
                                bx, by, bl, bt = buf
                                bx, by, bl, bt = bx.to(device), by.to(device), bl.to(device), bt.to(device)
                                for tid in bt.unique():
                                    mask = bt == tid
                                    r_logits = model(bx[mask], task_id=tid.item())
                                    loss = loss + F.cross_entropy(r_logits, by[mask]) / len(bt.unique())
                                    loss = loss + 0.5 * F.mse_loss(r_logits, bl[mask]) / len(bt.unique())
                        else:
                            buf = buffer.sample(64)
                            if buf is not None:
                                bx, by, bt = buf
                                bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                                for tid in bt.unique():
                                    mask = bt == tid
                                    loss = loss + F.cross_entropy(model(bx[mask], task_id=tid.item()), by[mask]) / len(bt.unique())

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Dynamic protection
                protector.scale_gradients(task_id)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_loss += loss.item()
                n_batches += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                stats = get_compression_stats(model)
                acc = evaluate_task(model, test_loader, task_id, device)

                # Compute avg risk (how much bit-depths have decreased)
                avg_risk = 0
                n_risk = 0
                for name in protector.reference_bitdepths:
                    for n, m in model.named_modules():
                        if n == name and hasattr(m, 'quantizer'):
                            curr = m.quantizer.get_channel_bit_depths().detach()
                            ref = protector.reference_bitdepths[name]
                            avg_risk += (ref - curr).clamp(min=0).mean().item()
                            n_risk += 1
                avg_risk = avg_risk / max(n_risk, 1)

                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_loss/n_batches:.4f} | "
                      f"Q: {stats['avg_bit_depth']:.2f} | Acc: {acc*100:.1f}% | "
                      f"Bits: {stats['compression_ratio']*100:.1f}% | "
                      f"Risk: {avg_risk:.4f}")

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
                        logits_out = model(x_raw.unsqueeze(0).to(device), task_id=task_id).squeeze(0).cpu()
                        samples.append((x_raw, local_y, logits_out, task_id))
                model.train()
                buffer.add_task_samples(samples)
            else:
                samples = benchmark.sample_for_replay(task_id, replay_per_task)
                buffer.add_task_samples(samples)

        # Evaluate
        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        stats = get_compression_stats(model)
        cl_metrics.update(task_id, all_accs, stats['total_bits'])

        for j, acc_val in enumerate(all_accs):
            print(f"  Task {j}: {acc_val*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl_metrics.backward_transfer(task_id)*100:.2f}%")
        print(f"  Bits: {stats['compression_ratio']*100:.1f}%")

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    cl_metrics.print_matrix()
    return cl_metrics, stats
