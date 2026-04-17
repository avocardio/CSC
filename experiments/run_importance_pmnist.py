"""Importance-only on Permuted MNIST (single-head)."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from data.permuted_mnist import PermutedMNIST
from data.replay_buffer import ReplayBuffer
from training.metrics import CLMetrics


class ImportanceOnlyMLP(nn.Module):
    """Single-head MLP with importance params but no quantization."""

    def __init__(self, init_bit_depth=8.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, 10)

        # Importance parameters per output unit of each hidden layer
        self.importance = nn.ParameterDict({
            'fc1': nn.Parameter(torch.full((256,), init_bit_depth)),
            'fc2': nn.Parameter(torch.full((256,), init_bit_depth)),
        })

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)

    def compute_avg_bitdepth(self):
        total = 0.0
        count = 0
        for name, b in self.importance.items():
            layer = getattr(self, name)
            total += (b.clamp(min=0) * layer.in_features).sum()
            count += layer.weight.numel()
        return total / max(count, 1)

    def scale_gradients(self, beta):
        for name, b in self.importance.items():
            bits = b.clamp(min=0).detach()
            layer = getattr(self, name)
            if layer.weight.grad is not None:
                scale = 1.0 / (1.0 + beta * bits.view(-1, 1))
                layer.weight.grad.data *= scale
            if b.grad is not None:
                b.grad.data *= 1.0 / (1.0 + beta * bits)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed); torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    bm = PermutedMNIST(num_tasks=args.num_tasks, batch_size=256, num_workers=4, seed=args.seed)
    model = ImportanceOnlyMLP().to(device)
    rb = ReplayBuffer(max_per_task=args.replay_per_task)
    cl = CLMetrics(args.num_tasks)

    for task_id in range(args.num_tasks):
        tl, vl = bm.get_task_dataloaders(task_id)

        imp_params = list(model.importance.parameters())
        imp_ids = {id(p) for p in imp_params}
        other = [p for p in model.parameters() if id(p) not in imp_ids]

        opt = torch.optim.AdamW([
            {'params': other, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': imp_params, 'lr': 0.1, 'eps': 1e-3, 'weight_decay': 0},
        ])

        for ep in range(args.epochs_per_task):
            model.train()
            for batch in tl:
                x, y = batch[0].to(device), batch[1].to(device)
                loss = F.cross_entropy(model(x), y)
                Q = model.compute_avg_bitdepth()
                loss = loss + args.gamma * Q
                if rb.size > 0:
                    rd = rb.sample(256)
                    if rd:
                        rx, ry = rd[0].to(device), rd[1].to(device)
                        loss = loss + F.cross_entropy(model(rx), ry)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                if args.beta > 0:
                    model.scale_gradients(args.beta)
                opt.step()

        rb.add_task_samples(bm.sample_for_replay(task_id, args.replay_per_task))

        model.eval()
        accs = []
        for t in range(task_id + 1):
            correct, total = 0, 0
            with torch.no_grad():
                for batch in bm.get_task_dataloaders(t)[1]:
                    x, y = batch[0].to(device), batch[1].to(device)
                    correct += (model(x).argmax(1) == y).sum().item()
                    total += y.size(0)
            accs.append(correct / total)
        cl.update(task_id, accs)

        if (task_id + 1) % max(1, args.num_tasks // 10) == 0 or task_id == args.num_tasks - 1:
            print(f"  Task {task_id}/{args.num_tasks-1}: Avg={cl.average_accuracy(task_id)*100:.1f}%")

    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'accuracy_matrix': cl.accuracy_matrix, 'avg_accuracy': avg, 'bwt': bwt},
               f'checkpoints/pmnist_imp_only_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
