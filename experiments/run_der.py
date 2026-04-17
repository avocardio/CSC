"""Run DER++ experiments with and without soft CSC."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity
from data.split_cifar100 import SplitCIFAR100
from training.der_hybrid import train_soft_csc_der


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0, help='soft protection strength')
    parser.add_argument('--alpha_der', type=float, default=0.5, help='DER++ MSE weight')
    parser.add_argument('--beta_der', type=float, default=1.0, help='DER++ CE weight')
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--adaptive_gamma', action='store_true')
    parser.add_argument('--rewire', action='store_true')
    parser.add_argument('--no_protection', action='store_true',
                        help='Disable soft protection (plain DER++ baseline)')
    parser.add_argument('--no_compression', action='store_true',
                        help='Disable compression (DER++ only baseline)')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'
    classes_per_task = 100 // args.num_tasks

    name_parts = [f"t{args.num_tasks}"]
    if args.no_compression:
        name_parts.append("der_only")
    elif args.no_protection:
        name_parts.append("csc_der")
    else:
        name_parts.append("soft_der")
    name_parts.append(f"r{args.replay_per_task}")
    name_parts.append(f"ader{args.alpha_der}")
    if args.adaptive_gamma:
        name_parts.append("adaptive")
    if args.rewire:
        name_parts.append("rewire")
    run_name = "_".join(name_parts)

    print(f"DER++ Experiment: {run_name}")

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    model = QuantizedResNet18(classes_per_task, args.num_tasks,
                              granularity=CompressionGranularity.CHANNEL).to(device)

    config = {
        'num_tasks': args.num_tasks,
        'epochs_per_task': args.epochs_per_task,
        'gamma': 0.0 if args.no_compression else args.gamma,
        'alpha_der': args.alpha_der,
        'beta_der': args.beta_der,
        'beta': 0.0 if args.no_protection else args.beta,
        'lr_weights': 1e-3,
        'lr_quant': 0.5,
        'eps_weights': 1e-5,
        'eps_quant': 1e-3,
        'weight_decay': 5e-4,
        'replay_per_task': args.replay_per_task,
        'replay_batch_size': 64,
        'bias_l1_weight': 0.01,
        'adaptive_gamma': args.adaptive_gamma,
        'gamma_base': args.gamma,
        'rewire': args.rewire,
    }

    cl_metrics, stats = train_soft_csc_der(model, benchmark, config, device)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
        'final_stats': stats,
        'config': config,
    }, f'checkpoints/der_{run_name}.pt')


if __name__ == '__main__':
    main()
