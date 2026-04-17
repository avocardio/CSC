"""Soft CSC + DER++ (Dark Experience Replay++) training loop.

DER++ stores logits alongside replay samples and adds MSE distillation loss.
Combined with soft protection for importance-based gradient scaling.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.quantization import (
    compute_average_bit_depth, get_compression_stats, get_quantizers,
    DifferentiableQuantizer,
)
from models.compression import apply_bias_l1_penalty
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.der_buffer import DERBuffer
from training.hybrid import SoftProtectionCSC, make_optimizer_adamw


def collect_der_samples(model, benchmark, task_id, n_samples, device):
    """Collect replay samples WITH current model logits for DER++."""
    task = benchmark.tasks[task_id]
    indices = task['train_indices']
    rng = __import__('numpy').random.RandomState(task_id)
    selected = rng.choice(len(indices), min(n_samples, len(indices)), replace=False)

    samples = []
    model.eval()
    with torch.no_grad():
        for i in selected:
            x, y = benchmark.train_dataset_raw[indices[i]]
            local_y = task['class_mapping'][y]
            # Get logits
            x_device = x.unsqueeze(0).to(device)
            logits = model(x_device, task_id=task_id).squeeze(0).cpu()
            samples.append((x, local_y, logits, task_id))
    model.train()
    return samples


def train_soft_csc_der(model, benchmark, config, device='cuda'):
    """Soft CSC + DER++ training loop."""
    model = model.to(device)
    num_tasks = config.get('num_tasks', 10)
    epochs_per_task = config.get('epochs_per_task', 50)
    gamma = config.get('gamma', 0.001)
    alpha_der = config.get('alpha_der', 0.5)  # DER++ logit distillation weight
    beta_der = config.get('beta_der', 1.0)    # DER++ CE replay weight
    beta_prot = config.get('beta', 1.0)       # soft protection strength
    replay_per_task = config.get('replay_per_task', 200)
    replay_batch_size = config.get('replay_batch_size', 64)
    bias_l1_weight = config.get('bias_l1_weight', 0.01)
    adaptive_gamma = config.get('adaptive_gamma', False)
    gamma_base = config.get('gamma_base', gamma)
    rewire = config.get('rewire', False)
    rewire_threshold = config.get('rewire_threshold', 0.5)

    der_buffer = DERBuffer(max_per_task=replay_per_task)
    cl_metrics = CLMetrics(num_tasks)
    protector = SoftProtectionCSC(model, beta=beta_prot, device=device)

    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_id} / {num_tasks-1}")
        print(f"{'='*60}")

        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        optimizer = make_optimizer_adamw(
            model,
            lr_weights=config.get('lr_weights', 1e-3),
            lr_quant=config.get('lr_quant', 0.5),
            eps_weights=config.get('eps_weights', 1e-5),
            eps_quant=config.get('eps_quant', 1e-3),
            weight_decay=config.get('weight_decay', 5e-4),
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_task * len(train_loader))

        # Adaptive gamma
        if adaptive_gamma:
            stats = get_compression_stats(model)
            remaining = stats['compression_ratio']
            gamma_eff = gamma_base * remaining
            print(f"  Adaptive gamma: base={gamma_base}, remaining={remaining:.3f}, effective={gamma_eff:.5f}")
        else:
            gamma_eff = gamma

        for epoch in range(epochs_per_task):
            model.train()
            epoch_losses = {'task': 0, 'der_mse': 0, 'der_ce': 0, 'comp': 0, 'total': 0}
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    task_loss = F.cross_entropy(logits, y)

                    Q = compute_average_bit_depth(model)
                    comp_loss = gamma_eff * Q
                    bias_loss = apply_bias_l1_penalty(model, bias_l1_weight)

                    loss = task_loss + comp_loss + bias_loss

                    # DER++ replay
                    if der_buffer.size > 0:
                        buf_data = der_buffer.sample(replay_batch_size)
                        if buf_data is not None:
                            bx, by, blogits, btids = buf_data
                            bx = bx.to(device)
                            by = by.to(device)
                            blogits = blogits.to(device)
                            btids = btids.to(device)

                            # Per-task forward for CE loss
                            der_ce_loss = torch.tensor(0.0, device=device)
                            der_mse_loss = torch.tensor(0.0, device=device)

                            for tid in btids.unique():
                                mask = btids == tid
                                buf_logits_current = model(bx[mask], task_id=tid.item())

                                # DER++ CE loss (using stored labels)
                                der_ce_loss += F.cross_entropy(buf_logits_current, by[mask])

                                # DER++ MSE loss (match stored logits)
                                der_mse_loss += F.mse_loss(buf_logits_current, blogits[mask])

                            n_unique = len(btids.unique())
                            der_ce_loss = der_ce_loss / n_unique
                            der_mse_loss = der_mse_loss / n_unique

                            loss = loss + beta_der * der_ce_loss + alpha_der * der_mse_loss

                            epoch_losses['der_ce'] += der_ce_loss.item()
                            epoch_losses['der_mse'] += der_mse_loss.item()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)

                # Soft protection
                protector.scale_gradients(task_id)

                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_losses['task'] += task_loss.item()
                epoch_losses['comp'] += Q.item()
                epoch_losses['total'] += loss.item()
                n_batches += 1

            # Connection rewiring
            if rewire and (epoch + 1) % 10 == 0:
                _rewire_dead_weights(model, rewire_threshold, optimizer)

            if (epoch + 1) % 10 == 0 or epoch == 0:
                stats = get_compression_stats(model)
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_losses['total']/n_batches:.4f} | "
                      f"DER_MSE: {epoch_losses['der_mse']/n_batches:.4f} | "
                      f"Acc: {acc*100:.1f}% | Bits: {stats['compression_ratio']*100:.1f}%")

        # Store DER++ samples (with logits)
        if replay_per_task > 0:
            samples = collect_der_samples(model, benchmark, task_id, replay_per_task, device)
            der_buffer.add_task_samples(samples)
            print(f"  DER++ buffer: {der_buffer.size} total")

        # Evaluate
        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        stats = get_compression_stats(model)
        cl_metrics.update(task_id, all_accs, stats['total_bits'])

        for j, acc in enumerate(all_accs):
            print(f"  Task {j}: {acc*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl_metrics.backward_transfer(task_id)*100:.2f}%")
        print(f"  Bits: {stats['compression_ratio']*100:.1f}%")

        os.makedirs('checkpoints', exist_ok=True)

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    cl_metrics.print_matrix()
    print(f"\nFinal bits: {stats['compression_ratio']*100:.1f}%")
    return cl_metrics, stats


def _rewire_dead_weights(model, threshold, optimizer):
    """Reinitialize weights with bit-depth below threshold."""
    count = 0
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer') and hasattr(module, 'conv'):
            q = module.quantizer
            bits = q.get_channel_bit_depths().detach()
            dead = bits < threshold
            if dead.any():
                n_dead = dead.sum().item()
                # Reinitialize dead channel weights
                with torch.no_grad():
                    w = module.conv.weight
                    nn.init.kaiming_normal_(w.data[dead], mode='fan_out', nonlinearity='relu')
                    # Reset bit-depth to mean of alive channels
                    alive_mean = bits[~dead].mean() if (~dead).any() else 4.0
                    q.bit_depth.data[dead] = alive_mean
                count += n_dead
    if count > 0:
        print(f"    Rewired {count} dead channels")
