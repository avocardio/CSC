"""Scaling gap experiment on Permuted MNIST (single-head)."""

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


class ScalingMLP(nn.Module):
    """Single-head MLP with optional importance parameters."""

    def __init__(self, has_importance=False, init_bd=8.0):
        super().__init__()
        self.fc1 = nn.Linear(784, 256)
        self.fc2 = nn.Linear(256, 256)
        self.head = nn.Linear(256, 10)
        self.has_importance = has_importance

        if has_importance:
            self.importance = nn.ParameterDict({
                'fc1': nn.Parameter(torch.full((256,), init_bd)),
                'fc2': nn.Parameter(torch.full((256,), init_bd)),
            })

    def forward(self, x):
        x = x.view(x.size(0), -1)
        return self.head(F.relu(self.fc2(F.relu(self.fc1(x)))))

    def compute_avg_importance(self):
        if not self.has_importance:
            return torch.tensor(0.0)
        total = sum(b.clamp(min=0).sum() for b in self.importance.values())
        count = sum(b.numel() for b in self.importance.values())
        return total / count


def train_pmnist(method, num_tasks, epochs, replay_per_task, gamma, beta, device, seed):
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)
    bm = PermutedMNIST(num_tasks=num_tasks, batch_size=256, num_workers=4, seed=seed)
    model = ScalingMLP(has_importance=(method == 'learned')).to(device)
    rb = ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)

    # Random scaling factors
    rng = torch.Generator(); rng.manual_seed(seed)
    random_scales = {
        'fc1': (torch.rand(256, generator=rng) * 7.9 + 0.1).to(device),
        'fc2': (torch.rand(256, generator=rng) * 7.9 + 0.1).to(device),
    }

    for task_id in range(num_tasks):
        tl, vl = bm.get_task_dataloaders(task_id)

        # Re-randomize for method D
        if method == 'rerandom':
            rng2 = torch.Generator(); rng2.manual_seed(seed + task_id * 1000)
            random_scales = {
                'fc1': (torch.rand(256, generator=rng2) * 7.9 + 0.1).to(device),
                'fc2': (torch.rand(256, generator=rng2) * 7.9 + 0.1).to(device),
            }

        if method == 'learned':
            imp_p = list(model.importance.parameters())
            imp_ids = {id(p) for p in imp_p}
            other = [p for p in model.parameters() if id(p) not in imp_ids]
            opt = torch.optim.AdamW([
                {'params': other, 'lr': 1e-3, 'weight_decay': 1e-4},
                {'params': imp_p, 'lr': 0.1, 'eps': 1e-3, 'weight_decay': 0},
            ])
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(epochs):
            model.train()
            for batch in tl:
                x, y = batch[0].to(device), batch[1].to(device)
                loss = F.cross_entropy(model(x), y)

                if method == 'learned':
                    loss = loss + gamma * model.compute_avg_importance()

                if rb.size > 0:
                    buf = rb.sample(256)
                    if buf:
                        rx, ry = buf[0].to(device), buf[1].to(device)
                        loss = loss + F.cross_entropy(model(rx), ry)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Apply scaling
                if method != 'replay':
                    for layer_name in ['fc1', 'fc2']:
                        layer = getattr(model, layer_name)
                        if layer.weight.grad is not None:
                            if method == 'learned':
                                factors = model.importance[layer_name].clamp(min=0).detach()
                            else:
                                factors = random_scales[layer_name]
                            scale = 1.0 / (1.0 + beta * factors.view(-1, 1))
                            layer.weight.grad.data *= scale

                    if method == 'learned':
                        for ln, bits in model.importance.items():
                            if bits.grad is not None:
                                f = bits.clamp(min=0).detach()
                                bits.grad.data *= 1.0 / (1.0 + beta * f)

                opt.step()

        rb.add_task_samples(bm.sample_for_replay(task_id, replay_per_task))

        model.eval()
        accs = []
        for t in range(task_id + 1):
            correct, total = 0, 0
            with torch.no_grad():
                for batch in bm.get_task_dataloaders(t)[1]:
                    bx, by = batch[0].to(device), batch[1].to(device)
                    correct += (model(bx).argmax(1) == by).sum().item()
                    total += by.size(0)
            accs.append(correct / total)
        cl.update(task_id, accs)

        if (task_id + 1) % max(1, num_tasks // 5) == 0 or task_id == num_tasks - 1:
            print(f"  [{method}] Task {task_id}/{num_tasks-1}: "
                  f"Avg={cl.average_accuracy(task_id)*100:.1f}%")

    return cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'replay', 'random_fixed', 'learned', 'rerandom'])
    args = parser.parse_args()

    device = 'cuda'
    methods = ['replay', 'random_fixed', 'learned', 'rerandom'] if args.method == 'all' else [args.method]
    results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"PMNIST: {method}, {args.num_tasks} tasks")
        print(f"{'='*50}")

        cl = train_pmnist(method, args.num_tasks, args.epochs_per_task,
                          args.replay_per_task, args.gamma, args.beta, device, args.seed)
        avg = cl.average_accuracy(args.num_tasks - 1)
        bwt = cl.backward_transfer(args.num_tasks - 1)
        results[method] = {'avg': avg, 'bwt': bwt}
        print(f"  Final: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    print(f"\n{'='*50}")
    print(f"SUMMARY (PMNIST, {args.num_tasks} tasks)")
    print(f"{'='*50}")
    for method, r in results.items():
        print(f"  {method:15s}: {r['avg']*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(results, f'checkpoints/scaling_gap_pmnist_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
