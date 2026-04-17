"""Run targeted CSC experiments."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from models.resnet import QuantizedResNet18
from models.quantization import CompressionGranularity
from data.split_cifar100 import SplitCIFAR100
from training.targeted_csc import train_targeted_csc


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta_base', type=float, default=0.5)
    parser.add_argument('--beta_risk', type=float, default=2.0)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--use_der', action='store_true')
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = 'cuda'
    classes_per_task = 100 // args.num_tasks

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    model = QuantizedResNet18(classes_per_task, args.num_tasks,
                              granularity=CompressionGranularity.CHANNEL).to(device)

    config = vars(args)
    print(f"Targeted CSC: g={args.gamma}, beta_base={args.beta_base}, "
          f"beta_risk={args.beta_risk}, DER++={args.use_der}")

    cl, stats = train_targeted_csc(model, benchmark, config, device)

    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%, Bits={stats['compression_ratio']*100:.1f}%")

    os.makedirs('checkpoints', exist_ok=True)
    name = f"targeted_g{args.gamma}_bb{args.beta_base}_br{args.beta_risk}"
    if args.use_der: name += "_der"
    torch.save({
        'accuracy_matrix': cl.accuracy_matrix,
        'summary': cl.summary(),
        'stats': stats,
    }, f'checkpoints/{name}.pt')


if __name__ == '__main__':
    main()
