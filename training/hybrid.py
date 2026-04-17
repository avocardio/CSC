"""Hybrid CSC + selective protection for continual learning.

Two variants:
A) Hard freeze: after each task, freeze weights with highest bit-depths (like PackNet
   but using learned importance from compression instead of magnitude pruning).
B) Soft protection: scale learning rates inversely with bit-depth. High bit-depth
   weights change slowly, low bit-depth weights remain malleable.

Both use the self-compression signal as a learned importance measure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from models.quantization import (
    compute_average_bit_depth, get_compression_stats, get_quantizers,
    DifferentiableQuantizer,
)
from models.compression import apply_bias_l1_penalty, remove_dead_channels
from training.metrics import evaluate_task, CLMetrics
from training.bn_utils import PerTaskBNTracker, evaluate_all_with_bn
from data.replay_buffer import ReplayBuffer


def make_optimizer_adamw(model, lr_weights=1e-3, lr_quant=0.5,
                         eps_weights=1e-5, eps_quant=1e-3, weight_decay=5e-4):
    """AdamW optimizer (decoupled weight decay) to avoid frozen weight drift."""
    quant_params = []
    weight_params = []
    quant_ids = set()

    for m in model.modules():
        if isinstance(m, DifferentiableQuantizer):
            for p in m.parameters():
                quant_ids.add(id(p))
                quant_params.append(p)

    for p in model.parameters():
        if id(p) not in quant_ids:
            weight_params.append(p)

    optimizer = torch.optim.AdamW([
        {'params': weight_params, 'lr': lr_weights, 'eps': eps_weights,
         'weight_decay': weight_decay},
        {'params': quant_params, 'lr': lr_quant, 'eps': eps_quant,
         'weight_decay': 0.0},
    ])
    return optimizer


class HardFreezeCSC:
    """Variant A: CSC with hard freezing of high-bit-depth weights after each task."""

    def __init__(self, model, freeze_ratio=0.5, device='cuda'):
        self.model = model
        self.device = device
        self.freeze_ratio = freeze_ratio

        # Frozen mask per parameter (True = frozen)
        self.frozen_masks = {}
        for name, param in model.named_parameters():
            if param.dim() >= 2 and 'heads.' not in name:
                self.frozen_masks[name] = torch.zeros_like(param, dtype=torch.bool)

        # Frozen values snapshot (for force-restore)
        self.frozen_values = {}

    def freeze_by_bitdepth(self, task_id):
        """After training task, freeze top freeze_ratio weights by bit-depth."""
        quantizers = get_quantizers(self.model)
        quant_map = {}
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer'):
                quant_map[name] = module

        total_frozen = 0
        total_eligible = 0

        for name in self.frozen_masks:
            # Find the corresponding quantized conv
            parent_name = '.'.join(name.split('.')[:-1])  # e.g., layer1.0.conv1.conv
            # Try to find the quantizer
            q_module = None
            for qname, qmod in quant_map.items():
                if name.startswith(qname):
                    q_module = qmod
                    break

            if q_module is None:
                continue

            param = dict(self.model.named_parameters())[name]
            free_mask = ~self.frozen_masks[name]
            n_free = free_mask.sum().item()
            if n_free == 0:
                continue

            total_eligible += n_free

            # Get channel bit depths and expand to weight level
            channel_bits = q_module.quantizer.get_channel_bit_depths().detach()
            # Expand to match weight shape: (O, I, H, W)
            if param.dim() == 4:
                bits_expanded = channel_bits.view(-1, 1, 1, 1).expand_as(param)
            elif param.dim() == 2:
                bits_expanded = channel_bits.view(-1, 1).expand_as(param)
            else:
                continue

            # Among free weights, find those with highest bit-depth
            free_bits = bits_expanded[free_mask]
            n_freeze = int(n_free * self.freeze_ratio)
            if n_freeze == 0:
                continue

            threshold = free_bits.kthvalue(n_free - n_freeze).values.item()
            new_freeze = free_mask & (bits_expanded >= threshold)

            self.frozen_masks[name] = self.frozen_masks[name] | new_freeze
            total_frozen += new_freeze.sum().item()

        # Snapshot frozen values
        self._snapshot_frozen()

        total_params = sum(m.numel() for m in self.frozen_masks.values())
        all_frozen = sum(m.sum().item() for m in self.frozen_masks.values())
        print(f"  Hard freeze: froze {total_frozen} new weights "
              f"(total frozen: {all_frozen}/{total_params} = {all_frozen/total_params*100:.1f}%)")

    def _snapshot_frozen(self):
        """Snapshot frozen weight values for force-restore."""
        self.frozen_values = {}
        for name, mask in self.frozen_masks.items():
            if mask.any():
                param = dict(self.model.named_parameters())[name]
                self.frozen_values[name] = param.data[mask].clone()

    def restore_frozen(self):
        """Force-restore frozen weights (call after each optimizer step)."""
        with torch.no_grad():
            for name, vals in self.frozen_values.items():
                param = dict(self.model.named_parameters())[name]
                param.data[self.frozen_masks[name]] = vals

    def zero_frozen_grads(self, task_id):
        """Zero gradients on frozen weights and other task heads."""
        for name, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if name in self.frozen_masks and self.frozen_masks[name].any():
                param.grad.data[self.frozen_masks[name]] = 0
            # Protect other task heads
            if 'heads.' in name:
                head_idx = int(name.split('heads.')[1].split('.')[0])
                if head_idx != task_id:
                    param.grad.data.fill_(0)

    def freeze_quant_params(self):
        """Freeze bit-depth params for frozen weights (channel-level)."""
        for name, module in self.model.named_modules():
            if hasattr(module, 'quantizer'):
                q = module.quantizer
                channel_frozen = self.frozen_masks.get(name + '.conv.weight',
                                  self.frozen_masks.get(name + '.weight', None))
                if channel_frozen is not None:
                    # If all weights in a channel are frozen, freeze its bit-depth too
                    frozen_per_channel = channel_frozen.view(channel_frozen.shape[0], -1).all(dim=1)
                    if frozen_per_channel.any() and q.bit_depth.grad is not None:
                        q.bit_depth.grad.data[frozen_per_channel] = 0
                        q.exponent.grad.data[frozen_per_channel] = 0


class SoftProtectionCSC:
    """Variant B: Scale learning rates inversely with bit-depth.

    effective_lr_i = base_lr / (1 + beta * b_i)
    or with relative scaling: base_lr / (1 + beta * (b_i / b_max))

    Implemented by scaling gradients: grad *= 1.0 / (1.0 + beta * b_i)
    """

    def __init__(self, model, beta=5.0, device='cuda', relative_scaling=False,
                 random_scaling=False):
        self.model = model
        self.device = device
        self.beta = beta
        self.relative_scaling = relative_scaling
        self.random_scaling = random_scaling

        # For random scaling: generate fixed random "importance" per channel
        # matching the distribution of bit-depths (uniform 0 to 8)
        if random_scaling:
            self.random_importance = {}
            for name, module in model.named_modules():
                if hasattr(module, 'quantizer'):
                    n_channels = module.quantizer.num_output_channels
                    self.random_importance[name] = torch.rand(n_channels, device=device) * 8.0

    def scale_gradients(self, task_id):
        """Scale gradients inversely with bit-depth for soft protection."""
        # Compute global b_max for relative scaling
        if self.relative_scaling:
            all_bits = []
            for name, module in self.model.named_modules():
                if hasattr(module, 'quantizer'):
                    all_bits.append(module.quantizer.get_channel_bit_depths().detach())
            b_max = torch.cat(all_bits).max().clamp(min=0.1) if all_bits else 1.0

        for name, module in self.model.named_modules():
            if not hasattr(module, 'quantizer'):
                continue
            q = module.quantizer
            channel_bits = q.get_channel_bit_depths().detach()

            if self.random_scaling:
                bits_for_scale = self.random_importance[name]
            elif self.relative_scaling:
                bits_for_scale = channel_bits / b_max
            else:
                bits_for_scale = channel_bits

            # Scale conv weight gradients
            if hasattr(module, 'conv') and module.conv.weight.grad is not None:
                weight = module.conv.weight
                if weight.dim() == 4:
                    scale = 1.0 / (1.0 + self.beta * bits_for_scale.view(-1, 1, 1, 1))
                elif weight.dim() == 2:
                    scale = 1.0 / (1.0 + self.beta * bits_for_scale.view(-1, 1))
                else:
                    continue
                weight.grad.data *= scale.expand_as(weight.grad)

            # Scale bit-depth gradients too (high bit-depth params resist change)
            if q.bit_depth.grad is not None:
                bit_scale = 1.0 / (1.0 + self.beta * bits_for_scale)
                q.bit_depth.grad.data *= bit_scale
            if q.exponent.grad is not None:
                q.exponent.grad.data *= bit_scale

        # Protect other task heads
        for pname, param in self.model.named_parameters():
            if param.grad is None:
                continue
            if 'heads.' in pname:
                head_idx = int(pname.split('heads.')[1].split('.')[0])
                if head_idx != task_id:
                    param.grad.data.fill_(0)


def train_hybrid_hard(model, benchmark, config, device='cuda'):
    """Train CSC with hard freezing of high-bit-depth weights."""
    model = model.to(device)
    num_tasks = config.get('num_tasks', 10)
    epochs_per_task = config.get('epochs_per_task', 50)
    gamma = config.get('gamma', 0.01)
    alpha = config.get('alpha', 1.0)
    replay_per_task = config.get('replay_per_task', 200)
    replay_batch_size = config.get('replay_batch_size', 64)
    freeze_ratio = config.get('freeze_ratio', 0.5)
    bias_l1_weight = config.get('bias_l1_weight', 0.01)

    replay_buffer = ReplayBuffer(max_per_task=replay_per_task)
    cl_metrics = CLMetrics(num_tasks)
    bn_tracker = PerTaskBNTracker()
    freezer = HardFreezeCSC(model, freeze_ratio=freeze_ratio, device=device)

    scaler = torch.amp.GradScaler('cuda')
    global_step = 0

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

        for epoch in range(epochs_per_task):
            model.train()
            epoch_losses = {'task': 0, 'comp': 0, 'replay': 0, 'total': 0}
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    task_loss = F.cross_entropy(logits, y)
                    Q = compute_average_bit_depth(model)
                    comp_loss = gamma * Q
                    bias_loss = apply_bias_l1_penalty(model, bias_l1_weight)
                    loss = task_loss + comp_loss + bias_loss

                    if replay_buffer.size > 0:
                        replay_data = replay_buffer.sample(replay_batch_size)
                        if replay_data is not None:
                            rx, ry, rtids = replay_data
                            rx, ry, rtids = rx.to(device), ry.to(device), rtids.to(device)
                            replay_loss = torch.tensor(0.0, device=device)
                            for tid in rtids.unique():
                                mask = rtids == tid
                                r_logits = model(rx[mask], task_id=tid.item())
                                replay_loss += F.cross_entropy(r_logits, ry[mask])
                            replay_loss = replay_loss / len(rtids.unique())
                            loss = loss + alpha * replay_loss
                            epoch_losses['replay'] += replay_loss.item()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                freezer.zero_frozen_grads(task_id)
                freezer.freeze_quant_params()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Force-restore frozen weights
                freezer.restore_frozen()

                epoch_losses['task'] += task_loss.item()
                epoch_losses['comp'] += Q.item()
                epoch_losses['total'] += loss.item()
                n_batches += 1
                global_step += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                stats = get_compression_stats(model)
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_losses['total']/n_batches:.4f} | "
                      f"Q: {epoch_losses['comp']/n_batches:.3f} | "
                      f"Acc: {acc*100:.1f}% | Bits: {stats['compression_ratio']*100:.1f}%")

        # Freeze high-bit-depth weights
        print(f"  Freezing top {freeze_ratio*100:.0f}% weights by bit-depth...")
        freezer.freeze_by_bitdepth(task_id)

        # Save BN and evaluate
        bn_tracker.save(model, task_id)
        all_accs = evaluate_all_with_bn(model, benchmark, task_id + 1, bn_tracker, device)
        stats = get_compression_stats(model)
        cl_metrics.update(task_id, all_accs, stats['total_bits'])

        for j, acc in enumerate(all_accs):
            print(f"  Task {j}: {acc*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl_metrics.backward_transfer(task_id)*100:.2f}%")
        print(f"  Bits: {stats['compression_ratio']*100:.1f}%")

        if replay_per_task > 0:
            samples = benchmark.sample_for_replay(task_id, replay_per_task)
            replay_buffer.add_task_samples(samples)

        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'accuracy_matrix': cl_metrics.accuracy_matrix,
            'stats': stats,
        }, f'checkpoints/hybrid_hard_task{task_id}.pt')

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    cl_metrics.print_matrix()
    print(f"\nFinal bits: {stats['compression_ratio']*100:.1f}%")
    return cl_metrics, stats


def train_hybrid_soft(model, benchmark, config, device='cuda'):
    """Train CSC with soft protection (LR scaling by bit-depth)."""
    model = model.to(device)
    num_tasks = config.get('num_tasks', 10)
    epochs_per_task = config.get('epochs_per_task', 50)
    gamma = config.get('gamma', 0.01)
    alpha = config.get('alpha', 1.0)
    replay_per_task = config.get('replay_per_task', 200)
    replay_batch_size = config.get('replay_batch_size', 64)
    beta = config.get('beta', 5.0)
    bias_l1_weight = config.get('bias_l1_weight', 0.01)

    relative_scaling = config.get('relative_scaling', False)
    random_scaling = config.get('random_scaling', False)

    replay_buffer = ReplayBuffer(max_per_task=replay_per_task)
    cl_metrics = CLMetrics(num_tasks)
    bn_tracker = PerTaskBNTracker()
    protector = SoftProtectionCSC(model, beta=beta, device=device,
                                  relative_scaling=relative_scaling,
                                  random_scaling=random_scaling)

    scaler = torch.amp.GradScaler('cuda')
    global_step = 0

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

        for epoch in range(epochs_per_task):
            model.train()
            epoch_losses = {'task': 0, 'comp': 0, 'replay': 0, 'total': 0}
            n_batches = 0

            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    task_loss = F.cross_entropy(logits, y)
                    Q = compute_average_bit_depth(model)
                    comp_loss = gamma * Q
                    bias_loss = apply_bias_l1_penalty(model, bias_l1_weight)
                    loss = task_loss + comp_loss + bias_loss

                    if replay_buffer.size > 0:
                        replay_data = replay_buffer.sample(replay_batch_size)
                        if replay_data is not None:
                            rx, ry, rtids = replay_data
                            rx, ry, rtids = rx.to(device), ry.to(device), rtids.to(device)
                            replay_loss = torch.tensor(0.0, device=device)
                            for tid in rtids.unique():
                                mask = rtids == tid
                                r_logits = model(rx[mask], task_id=tid.item())
                                replay_loss += F.cross_entropy(r_logits, ry[mask])
                            replay_loss = replay_loss / len(rtids.unique())
                            loss = loss + alpha * replay_loss
                            epoch_losses['replay'] += replay_loss.item()

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                # Apply soft protection: scale gradients by bit-depth
                protector.scale_gradients(task_id)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_losses['task'] += task_loss.item()
                epoch_losses['comp'] += Q.item()
                epoch_losses['total'] += loss.item()
                n_batches += 1
                global_step += 1

            if (epoch + 1) % 10 == 0 or epoch == 0:
                stats = get_compression_stats(model)
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_losses['total']/n_batches:.4f} | "
                      f"Q: {epoch_losses['comp']/n_batches:.3f} | "
                      f"Acc: {acc*100:.1f}% | Bits: {stats['compression_ratio']*100:.1f}%")

        # Evaluate: soft protection doesn't freeze weights, so use standard eval
        # (per-task BN swap would hurt because weights have changed)
        from training.metrics import evaluate_all_tasks
        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        stats = get_compression_stats(model)
        cl_metrics.update(task_id, all_accs, stats['total_bits'])

        for j, acc in enumerate(all_accs):
            print(f"  Task {j}: {acc*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  BWT: {cl_metrics.backward_transfer(task_id)*100:.2f}%")
        print(f"  Bits: {stats['compression_ratio']*100:.1f}%")

        if replay_per_task > 0:
            samples = benchmark.sample_for_replay(task_id, replay_per_task)
            replay_buffer.add_task_samples(samples)

        os.makedirs('checkpoints', exist_ok=True)
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'accuracy_matrix': cl_metrics.accuracy_matrix,
            'stats': stats,
        }, f'checkpoints/hybrid_soft_task{task_id}.pt')

    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    cl_metrics.print_matrix()
    print(f"\nFinal bits: {stats['compression_ratio']*100:.1f}%")
    return cl_metrics, stats
