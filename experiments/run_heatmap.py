"""Generate bit-depth evolution heatmap for the best CSC configuration."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity, get_compression_stats, get_quantizers
from data.split_cifar100 import SplitCIFAR100
from data.replay_buffer import ReplayBuffer
from training.hybrid import train_hybrid_soft, SoftProtectionCSC, make_optimizer_adamw
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from training.bn_utils import PerTaskBNTracker
from analysis.bitdepth_heatmap import BitDepthTracker
from models.compression import apply_bias_l1_penalty
from models.quantization import compute_average_bit_depth
import torch.nn.functional as F


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--record_interval', type=int, default=5,
                        help='Record bit-depths every N epochs')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=128,
                               num_workers=8, seed=args.seed)
    model = QuantizedResNet18(10, args.num_tasks,
                              granularity=CompressionGranularity.CHANNEL).to(device)

    protector = SoftProtectionCSC(model, beta=args.beta, device=device)
    replay_buffer = ReplayBuffer(max_per_task=args.replay_per_task)
    bd_tracker = BitDepthTracker()

    global_step = 0
    for task_id in range(args.num_tasks):
        print(f"\nTASK {task_id}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        optimizer = make_optimizer_adamw(model)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=args.epochs_per_task * len(train_loader))
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(args.epochs_per_task):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)
                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)
                    Q = compute_average_bit_depth(model)
                    loss = loss + args.gamma * Q + apply_bias_l1_penalty(model, 0.01)
                    if replay_buffer.size > 0:
                        rd = replay_buffer.sample(64)
                        if rd is not None:
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
                global_step += 1

            # Record bit-depths periodically
            if (epoch + 1) % args.record_interval == 0:
                bd_tracker.record(global_step, task_id, model)

            if (epoch + 1) % 10 == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                stats = get_compression_stats(model)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%, Bits={stats['compression_ratio']*100:.1f}%")

        samples = benchmark.sample_for_replay(task_id, args.replay_per_task)
        replay_buffer.add_task_samples(samples)

    # Generate heatmaps
    os.makedirs('figures', exist_ok=True)
    bd_tracker.plot_heatmap(save_path='figures/bitdepth_heatmap.png')
    bd_tracker.plot_layer_summary(save_path='figures/bitdepth_layers.png')
    print("\nHeatmaps saved to figures/")


if __name__ == '__main__':
    main()
