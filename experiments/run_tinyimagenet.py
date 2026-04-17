"""Run Soft CSC and replay-only on Split TinyImageNet."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity
from data.split_tinyimagenet import SplitTinyImageNet
from training.hybrid import train_hybrid_soft
from baselines.finetune import SimpleResNet18
from baselines.replay_only import train_replay


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['soft_csc', 'replay_only'])
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    classes_per_task = 200 // args.num_tasks

    print(f"Split TinyImageNet: {args.method}")
    print(f"  {args.num_tasks} tasks x {classes_per_task} classes")

    benchmark = SplitTinyImageNet(
        num_tasks=args.num_tasks, batch_size=args.batch_size, seed=args.seed)

    if args.method == 'soft_csc':
        model = QuantizedResNet18(
            num_classes_per_task=classes_per_task,
            num_tasks=args.num_tasks,
            granularity=CompressionGranularity.CHANNEL,
            image_size=64).to(device)

        config = {
            'num_tasks': args.num_tasks,
            'epochs_per_task': args.epochs_per_task,
            'gamma': args.gamma,
            'alpha': 1.0,
            'lr_weights': 1e-3,
            'lr_quant': 0.5,
            'eps_weights': 1e-5,
            'eps_quant': 1e-3,
            'weight_decay': 5e-4,
            'replay_per_task': args.replay_per_task,
            'replay_batch_size': 64,
            'bias_l1_weight': 0.01,
            'beta': args.beta,
            'relative_scaling': False,
        }
        cl_metrics, stats = train_hybrid_soft(model, benchmark, config, device)

    elif args.method == 'replay_only':
        # Need to override SimpleResNet18 for 64x64
        model = SimpleResNet18(classes_per_task, args.num_tasks, image_size=64).to(device)
        config = {
            'num_tasks': args.num_tasks,
            'epochs_per_task': args.epochs_per_task,
            'replay_per_task': args.replay_per_task,
            'lr': 1e-3,
            'batch_size': args.batch_size,
        }
        cl_metrics = train_replay(benchmark, config, device, model=model)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
    }, f'checkpoints/tinyimagenet_{args.method}.pt')


if __name__ == '__main__':
    main()
