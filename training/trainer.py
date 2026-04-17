"""Main CSC training loop for continual learning."""

import torch
import torch.nn.functional as F
import time
import wandb
from models.quantization import (
    compute_average_bit_depth, get_compression_stats, get_quantizers,
    DifferentiableQuantizer,
)
from models.compression import apply_bias_l1_penalty, remove_dead_channels
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics


def make_optimizer(model, lr_weights=1e-3, lr_quant=0.5,
                   eps_weights=1e-5, eps_quant=1e-3, weight_decay=5e-4):
    """Create Adam optimizer with separate param groups for weights and quantization params."""
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

    optimizer = torch.optim.Adam([
        {'params': weight_params, 'lr': lr_weights, 'eps': eps_weights,
         'weight_decay': weight_decay},
        {'params': quant_params, 'lr': lr_quant, 'eps': eps_quant,
         'weight_decay': 0.0},
    ])
    return optimizer


def train_single_task_compression(model, train_loader, test_loader, config, device='cuda'):
    """Train self-compression on a single task (Phase 1 validation).

    Args:
        model: QuantizedResNet18 (with single_head=True or task_id=0)
        train_loader, test_loader: data loaders
        config: dict with gamma, epochs, etc.
        device: cuda/cpu
    """
    model = model.to(device)
    model.train()

    optimizer = make_optimizer(
        model,
        lr_weights=config.get('lr_weights', 1e-3),
        lr_quant=config.get('lr_quant', 0.5),
        eps_weights=config.get('eps_weights', 1e-5),
        eps_quant=config.get('eps_quant', 1e-3),
        weight_decay=config.get('weight_decay', 5e-4),
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=10)

    gamma = config.get('gamma', 0.01)
    epochs = config.get('epochs', 200)
    removal_interval = config.get('removal_interval', 50)
    bias_l1_weight = config.get('bias_l1_weight', 0.01)
    use_wandb = config.get('use_wandb', False)

    scaler = torch.amp.GradScaler('cuda')

    best_acc = 0.0
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        epoch_task_loss = 0.0
        epoch_comp_loss = 0.0
        n_batches = 0

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            with torch.amp.autocast('cuda'):
                logits = model(x, task_id=0)
                task_loss = F.cross_entropy(logits, y)
                Q = compute_average_bit_depth(model)
                comp_loss = gamma * Q
                bias_loss = apply_bias_l1_penalty(model, bias_l1_weight)
                loss = task_loss + comp_loss + bias_loss

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            epoch_loss += loss.item()
            epoch_task_loss += task_loss.item()
            epoch_comp_loss += Q.item()
            n_batches += 1

        # Periodically try removing dead channels
        if (epoch + 1) % removal_interval == 0:
            removed = remove_dead_channels(model, optimizer)
            if removed > 0:
                print(f"  Removed {removed} dead channels at epoch {epoch+1}")

        # Evaluate
        avg_loss = epoch_loss / n_batches
        scheduler.step(avg_loss)

        stats = get_compression_stats(model)
        acc = evaluate_task(model, test_loader, 0, device)
        best_acc = max(best_acc, acc)

        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch {epoch+1:3d} | Loss: {avg_loss:.4f} | TaskLoss: {epoch_task_loss/n_batches:.4f} | "
                  f"Q: {epoch_comp_loss/n_batches:.3f} | Acc: {acc*100:.2f}% | "
                  f"Bits: {stats['compression_ratio']*100:.1f}% | "
                  f"Channels: {stats['total_channels']-stats['zero_channels']}/{stats['total_channels']}")

        if use_wandb:
            wandb.log({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'task_loss': epoch_task_loss / n_batches,
                'avg_bit_depth': epoch_comp_loss / n_batches,
                'accuracy': acc,
                'best_accuracy': best_acc,
                'compression_ratio': stats['compression_ratio'],
                'channels_remaining_pct': stats['channels_remaining_pct'],
                'total_bits': stats['total_bits'],
            })

    return best_acc, stats


def train_csc_continual(model, benchmark, config, device='cuda'):
    """Full CSC continual learning training loop.

    Args:
        model: QuantizedResNet18 with multi-head
        benchmark: SplitCIFAR100 or similar
        config: training configuration dict
        device: cuda/cpu
    """
    from data.replay_buffer import ReplayBuffer

    model = model.to(device)
    num_tasks = config.get('num_tasks', 10)
    epochs_per_task = config.get('epochs_per_task', 50)
    gamma = config.get('gamma', 0.01)
    alpha = config.get('alpha', 1.0)
    replay_per_task = config.get('replay_per_task', 200)
    replay_batch_size = config.get('replay_batch_size', 64)
    removal_interval = config.get('removal_interval', 20)
    bias_l1_weight = config.get('bias_l1_weight', 0.01)
    use_wandb = config.get('use_wandb', False)

    from training.bn_utils import PerTaskBNTracker, evaluate_all_with_bn

    replay_buffer = ReplayBuffer(max_per_task=replay_per_task)
    cl_metrics = CLMetrics(num_tasks)
    bn_tracker = PerTaskBNTracker()

    scaler = torch.amp.GradScaler('cuda')
    global_step = 0

    for task_id in range(num_tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_id} / {num_tasks-1}")
        print(f"{'='*60}")

        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Fresh optimizer per task to avoid stale Adam state and LR issues
        optimizer = make_optimizer(
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
                    # Current task loss
                    logits = model(x, task_id=task_id)
                    task_loss = F.cross_entropy(logits, y)

                    # Compression loss
                    Q = compute_average_bit_depth(model)
                    comp_loss = gamma * Q

                    # Bias L1
                    bias_loss = apply_bias_l1_penalty(model, bias_l1_weight)

                    loss = task_loss + comp_loss + bias_loss

                    # Replay loss
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
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                epoch_losses['task'] += task_loss.item()
                epoch_losses['comp'] += Q.item()
                epoch_losses['total'] += loss.item()
                n_batches += 1
                global_step += 1

            # Channel removal check
            if (epoch + 1) % removal_interval == 0:
                removed = remove_dead_channels(model, optimizer)
                if removed > 0:
                    print(f"  Removed {removed} dead channels at epoch {epoch+1}")

            # Logging
            if (epoch + 1) % 10 == 0 or epoch == 0:
                stats = get_compression_stats(model)
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1:3d} | Loss: {epoch_losses['total']/n_batches:.4f} | "
                      f"Task: {epoch_losses['task']/n_batches:.4f} | "
                      f"Q: {epoch_losses['comp']/n_batches:.3f} | "
                      f"Replay: {epoch_losses['replay']/n_batches:.4f} | "
                      f"Acc: {acc*100:.1f}% | Bits: {stats['compression_ratio']*100:.1f}%")

                if use_wandb:
                    wandb.log({
                        'task_id': task_id,
                        'epoch': epoch + 1,
                        'global_step': global_step,
                        'task_loss': epoch_losses['task'] / n_batches,
                        'compression_Q': epoch_losses['comp'] / n_batches,
                        'replay_loss': epoch_losses['replay'] / n_batches,
                        'total_loss': epoch_losses['total'] / n_batches,
                        'current_task_acc': acc,
                        'compression_ratio': stats['compression_ratio'],
                        'bits_remaining_pct': stats['compression_ratio'] * 100,
                        'channels_remaining_pct': stats['channels_remaining_pct'],
                    })

        # End of task: evaluate all tasks
        # Note: per-task BN swap is NOT used for CSC/replay because weights
        # continue evolving across tasks. BN swap only correct when weights are
        # frozen (PackNet). For shared-weight methods, current BN is correct.
        print(f"\nEvaluating all tasks after Task {task_id}...")
        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        stats = get_compression_stats(model)
        cl_metrics.update(task_id, all_accs, stats['total_bits'])

        for j, acc in enumerate(all_accs):
            print(f"  Task {j}: {acc*100:.1f}%")
        print(f"  Avg Accuracy: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        if task_id > 0:
            print(f"  Backward Transfer: {cl_metrics.backward_transfer(task_id)*100:.2f}%")
        print(f"  Bits remaining: {stats['compression_ratio']*100:.1f}%")

        if use_wandb:
            log_dict = {
                f'task_{j}_acc_after_task_{task_id}': acc
                for j, acc in enumerate(all_accs)
            }
            log_dict['avg_accuracy'] = cl_metrics.average_accuracy(task_id)
            log_dict['backward_transfer'] = cl_metrics.backward_transfer(task_id)
            log_dict['bits_remaining_pct_end_of_task'] = stats['compression_ratio'] * 100
            wandb.log(log_dict)

        # Store replay samples
        if replay_per_task > 0:
            samples = benchmark.sample_for_replay(task_id, replay_per_task)
            replay_buffer.add_task_samples(samples)
            print(f"  Replay buffer: {replay_buffer.size} total samples")

        # Save per-task checkpoint for crash resilience
        import os
        os.makedirs('checkpoints', exist_ok=True)
        ckpt_path = f'checkpoints/csc_task{task_id}.pt'
        torch.save({
            'task_id': task_id,
            'model_state_dict': model.state_dict(),
            'accuracy_matrix': cl_metrics.accuracy_matrix,
            'bits_history': cl_metrics.bits_history,
            'replay_data': replay_buffer.data,
            'stats': stats,
        }, ckpt_path)
        print(f"  Checkpoint saved: {ckpt_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")
    cl_metrics.print_matrix()
    summary = cl_metrics.summary()
    print(f"\nFinal bits remaining: {stats['compression_ratio']*100:.1f}%")

    return cl_metrics, stats
