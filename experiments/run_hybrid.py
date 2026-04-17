"""Run hybrid CSC experiments: hard freeze and soft protection variants."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity
from data.split_cifar100 import SplitCIFAR100
from training.hybrid import train_hybrid_hard, train_hybrid_soft


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--variant', type=str, required=True, choices=['hard', 'soft'])
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    # Hard freeze params
    parser.add_argument('--freeze_ratio', type=float, default=0.5)
    # Soft protection params
    parser.add_argument('--beta', type=float, default=5.0)
    parser.add_argument('--relative_scaling', action='store_true')
    parser.add_argument('--random_scaling', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    classes_per_task = 100 // args.num_tasks

    print(f"Hybrid CSC ({args.variant}) on Split CIFAR-100")
    if args.variant == 'hard':
        print(f"  freeze_ratio={args.freeze_ratio}")
    else:
        print(f"  beta={args.beta}")
    print(f"  gamma={args.gamma}, replay={args.replay_per_task}/task")

    benchmark = SplitCIFAR100(
        num_tasks=args.num_tasks, batch_size=args.batch_size,
        num_workers=8, seed=args.seed)

    model = QuantizedResNet18(
        num_classes_per_task=classes_per_task,
        num_tasks=args.num_tasks,
        granularity=CompressionGranularity.CHANNEL,
        init_bit_depth=8.0)

    config = {
        'num_tasks': args.num_tasks,
        'epochs_per_task': args.epochs_per_task,
        'gamma': args.gamma,
        'alpha': args.alpha,
        'lr_weights': 1e-3,
        'lr_quant': 0.5,
        'eps_weights': 1e-5,
        'eps_quant': 1e-3,
        'weight_decay': 5e-4,
        'replay_per_task': args.replay_per_task,
        'replay_batch_size': 64,
        'bias_l1_weight': 0.01,
        'freeze_ratio': args.freeze_ratio,
        'beta': args.beta,
        'relative_scaling': args.relative_scaling,
        'random_scaling': args.random_scaling,
    }

    if args.variant == 'hard':
        cl_metrics, stats = train_hybrid_hard(model, benchmark, config, device)
        name = f"hybrid_hard_f{args.freeze_ratio}"
    else:
        cl_metrics, stats = train_hybrid_soft(model, benchmark, config, device)
        name = f"hybrid_soft_b{args.beta}"

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
        'final_stats': stats,
        'config': config,
    }, f'checkpoints/{name}.pt')


if __name__ == '__main__':
    main()
