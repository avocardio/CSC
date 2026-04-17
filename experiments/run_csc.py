"""Run CSC continual learning experiment on Split CIFAR-100."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
import wandb
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity
from data.split_cifar100 import SplitCIFAR100
from training.trainer import train_csc_continual


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--alpha', type=float, default=1.0)
    parser.add_argument('--granularity', type=str, default='channel',
                        choices=['channel', 'group', 'weight'])
    parser.add_argument('--group_size', type=int, default=16)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--replay_batch_size', type=int, default=64)
    parser.add_argument('--init_bit_depth', type=float, default=8.0)
    parser.add_argument('--wandb', action='store_true')
    parser.add_argument('--wandb_project', type=str, default='csc-continual')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--no_compression', action='store_true',
                        help='Disable compression (replay-only baseline)')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    granularity = CompressionGranularity(args.granularity)
    classes_per_task = 100 // args.num_tasks

    run_name = f"csc_t{args.num_tasks}_g{args.gamma}_r{args.replay_per_task}_{args.granularity}"
    if args.no_compression:
        run_name = f"replay_only_t{args.num_tasks}_r{args.replay_per_task}"

    if args.wandb:
        wandb.init(project=args.wandb_project, config=vars(args), name=run_name)

    print(f"CSC Continual Learning on Split CIFAR-100")
    print(f"  {args.num_tasks} tasks x {classes_per_task} classes")
    print(f"  gamma={args.gamma}, alpha={args.alpha}")
    print(f"  granularity={args.granularity}, replay={args.replay_per_task}/task")
    print(f"  device={device}")

    # Data
    benchmark = SplitCIFAR100(
        num_tasks=args.num_tasks,
        batch_size=args.batch_size,
        num_workers=8,
        seed=args.seed,
    )

    # Model
    model = QuantizedResNet18(
        num_classes_per_task=classes_per_task,
        num_tasks=args.num_tasks,
        granularity=granularity,
        group_size=args.group_size,
        init_bit_depth=args.init_bit_depth,
    )
    print(f"  Model params: {sum(p.numel() for p in model.parameters()):,}")

    config = {
        'num_tasks': args.num_tasks,
        'epochs_per_task': args.epochs_per_task,
        'gamma': 0.0 if args.no_compression else args.gamma,
        'alpha': args.alpha,
        'lr_weights': 1e-3,
        'lr_quant': 0.5,
        'eps_weights': 1e-5,
        'eps_quant': 1e-3,
        'weight_decay': 5e-4,
        'replay_per_task': args.replay_per_task,
        'replay_batch_size': args.replay_batch_size,
        'removal_interval': 20,
        'bias_l1_weight': 0.01,
        'use_wandb': args.wandb,
    }

    cl_metrics, final_stats = train_csc_continual(model, benchmark, config, device)

    # Save
    os.makedirs('checkpoints', exist_ok=True)
    summary = cl_metrics.summary()
    torch.save({
        'model_state_dict': model.state_dict(),
        'config': config,
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': summary,
        'final_stats': final_stats,
    }, f'checkpoints/{run_name}.pt')

    if args.wandb:
        wandb.finish()


if __name__ == '__main__':
    main()
