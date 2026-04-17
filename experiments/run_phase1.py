"""Phase 1: Validate self-compression on single CIFAR-10 task.

Reproduces the core result from Csefalvay & Imber (2023):
Train ResNet-18 with differentiable quantization on CIFAR-10,
verify compression-accuracy tradeoff.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import yaml
import wandb
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity, get_compression_stats
from data.split_cifar100 import SplitCIFAR10
from training.trainer import train_single_task_compression


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--epochs', type=int, default=200)
    parser.add_argument('--granularity', type=str, default='channel',
                        choices=['channel', 'group', 'weight'])
    parser.add_argument('--group_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--init_bit_depth', type=float, default=8.0)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='csc-phase1')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    granularity = CompressionGranularity(args.granularity)

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args),
                   name=f"phase1_g{args.gamma}_{args.granularity}")

    print(f"Phase 1: Self-Compression on CIFAR-10")
    print(f"  gamma={args.gamma}, granularity={args.granularity}")
    print(f"  epochs={args.epochs}, batch_size={args.batch_size}")
    print(f"  device={device}")

    # Data
    cifar10 = SplitCIFAR10(batch_size=args.batch_size)
    train_loader, test_loader = cifar10.get_dataloaders()

    # Model
    model = QuantizedResNet18(
        num_classes_per_task=10, num_tasks=1,
        granularity=granularity,
        group_size=args.group_size,
        init_bit_depth=args.init_bit_depth,
        single_head=False,
    )

    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    config = {
        'gamma': args.gamma,
        'epochs': args.epochs,
        'lr_weights': 1e-3,
        'lr_quant': 0.5,
        'eps_weights': 1e-5,
        'eps_quant': 1e-3,
        'weight_decay': 5e-4,
        'removal_interval': 50,
        'bias_l1_weight': 0.01,
        'use_wandb': args.wandb,
    }

    best_acc, final_stats = train_single_task_compression(
        model, train_loader, test_loader, config, device)

    print(f"\nFinal Results:")
    print(f"  Best Accuracy: {best_acc*100:.2f}%")
    print(f"  Compression Ratio: {final_stats['compression_ratio']*100:.2f}%")
    print(f"  Avg Bit Depth: {final_stats['avg_bit_depth']:.3f}")
    print(f"  Channels Remaining: {final_stats['total_channels']-final_stats['zero_channels']}/{final_stats['total_channels']}")

    # Save model
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'best_acc': best_acc,
        'stats': final_stats,
    }, f'checkpoints/phase1_g{args.gamma}_{args.granularity}.pt')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
