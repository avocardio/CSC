"""Loss functions for Continual Self-Compression."""

import torch
import torch.nn.functional as F
from models.quantization import compute_average_bit_depth
from models.compression import apply_bias_l1_penalty


def csc_loss(model, x, y, task_id, replay_buffer=None, gamma=0.01,
             alpha=1.0, replay_batch_size=64, bias_l1_weight=0.01,
             device='cuda'):
    """Compute the full CSC loss.

    L = L_task + gamma * Q + alpha * L_replay + bias_l1

    Args:
        model: QuantizedResNet18
        x, y: current task batch
        task_id: current task id
        replay_buffer: ReplayBuffer or None
        gamma: compression factor
        alpha: replay loss weight
        replay_batch_size: number of replay samples per step
        bias_l1_weight: weight for L1 penalty on zero-bit channel biases
        device: cuda/cpu

    Returns:
        total_loss, loss_dict with individual components
    """
    # Task loss
    logits = model(x, task_id=task_id)
    task_loss = F.cross_entropy(logits, y)

    # Compression loss
    Q = compute_average_bit_depth(model)
    compression_loss = gamma * Q

    # Bias L1 penalty for zero-bit channels
    bias_loss = apply_bias_l1_penalty(model, bias_l1_weight)

    total_loss = task_loss + compression_loss + bias_loss

    loss_dict = {
        'task_loss': task_loss.item(),
        'compression_loss': compression_loss.item(),
        'Q': Q.item(),
        'bias_l1': bias_loss.item(),
    }

    # Replay loss
    if replay_buffer is not None and replay_buffer.size > 0:
        replay_data = replay_buffer.sample(replay_batch_size)
        if replay_data is not None:
            rx, ry, rtids = replay_data
            rx, ry, rtids = rx.to(device), ry.to(device), rtids.to(device)

            # Compute replay loss per task
            replay_loss = torch.tensor(0.0, device=device)
            unique_tasks = rtids.unique()
            for tid in unique_tasks:
                mask = rtids == tid
                r_logits = model(rx[mask], task_id=tid.item())
                replay_loss += F.cross_entropy(r_logits, ry[mask])
            replay_loss = replay_loss / len(unique_tasks)

            total_loss = total_loss + alpha * replay_loss
            loss_dict['replay_loss'] = replay_loss.item()
    else:
        loss_dict['replay_loss'] = 0.0

    loss_dict['total_loss'] = total_loss.item()
    return total_loss, loss_dict
