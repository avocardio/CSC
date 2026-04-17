"""Permuted MNIST scaling experiment (single-head, standard CL benchmark).

All methods use single-head (shared 10-class output).
This is the standard setup for Permuted MNIST in CL literature.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from data.permuted_mnist import PermutedMNIST
from data.replay_buffer import ReplayBuffer
from training.metrics import CLMetrics
from models.quantization import DifferentiableQuantizer, CompressionGranularity
from models.mlp import QuantizedLinear


class SingleHeadQuantizedMLP(nn.Module):
    """Single-head MLP with differentiable quantization for PMNIST."""

    def __init__(self, init_bit_depth=8.0):
        super().__init__()
        self.fc1 = QuantizedLinear(784, 256, granularity=CompressionGranularity.CHANNEL,
                                   init_bit_depth=init_bit_depth)
        self.fc2 = QuantizedLinear(256, 256, granularity=CompressionGranularity.CHANNEL,
                                   init_bit_depth=init_bit_depth)
        self.head = nn.Linear(256, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.head(x)


class SingleHeadMLP(nn.Module):
    """Single-head plain MLP for baselines."""

    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Flatten(), nn.Linear(784, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 10))

    def forward(self, x):
        return self.net(x)


def eval_task(model, test_loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for batch in test_loader:
            x, y = batch[0].to(device), batch[1].to(device)
            correct += (model(x).argmax(1) == y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total


def eval_all(model, benchmark, num_tasks, device):
    return [eval_task(model, benchmark.get_task_dataloaders(t)[1], device)
            for t in range(num_tasks)]


def train_soft_csc(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config.get('epochs_per_task', 5)
    gamma = config.get('gamma', 0.001)
    beta = config.get('beta', 1.0)
    replay_per_task = config.get('replay_per_task', 200)

    model = SingleHeadQuantizedMLP().to(device)
    rb = ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)

    for task_id in range(num_tasks):
        tl, vl = benchmark.get_task_dataloaders(task_id)

        quant_ids = set()
        qp, wp = [], []
        for m in model.modules():
            if isinstance(m, DifferentiableQuantizer):
                for p in m.parameters():
                    quant_ids.add(id(p)); qp.append(p)
        for p in model.parameters():
            if id(p) not in quant_ids: wp.append(p)

        opt = torch.optim.AdamW([
            {'params': wp, 'lr': 1e-3, 'weight_decay': 1e-4},
            {'params': qp, 'lr': 0.1, 'eps': 1e-3, 'weight_decay': 0},
        ])

        for ep in range(epochs):
            model.train()
            for batch in tl:
                x, y = batch[0].to(device), batch[1].to(device)
                loss = F.cross_entropy(model(x), y)

                # Compression loss
                from models.quantization import compute_average_bit_depth
                Q = compute_average_bit_depth(model)
                loss = loss + gamma * Q

                # Replay
                if rb.size > 0:
                    rd = rb.sample(256)
                    if rd is not None:
                        rx, ry = rd[0].to(device), rd[1].to(device)
                        loss = loss + F.cross_entropy(model(rx), ry)

                opt.zero_grad(set_to_none=True)
                loss.backward()

                # Soft protection
                for name, module in model.named_modules():
                    if hasattr(module, 'quantizer'):
                        q = module.quantizer
                        cb = q.get_channel_bit_depths().detach()
                        if hasattr(module, 'linear') and module.linear.weight.grad is not None:
                            s = 1.0 / (1.0 + beta * cb.view(-1, 1))
                            module.linear.weight.grad.data *= s
                        if q.bit_depth.grad is not None:
                            bs = 1.0 / (1.0 + beta * cb)
                            q.bit_depth.grad.data *= bs
                        if q.exponent.grad is not None:
                            q.exponent.grad.data *= bs

                opt.step()

        rb.add_task_samples(benchmark.sample_for_replay(task_id, replay_per_task))
        accs = eval_all(model, benchmark, task_id + 1, device)
        cl.update(task_id, accs)

        if (task_id + 1) % max(1, num_tasks // 10) == 0 or task_id == num_tasks - 1:
            print(f"  Task {task_id}/{num_tasks-1}: Avg={cl.average_accuracy(task_id)*100:.1f}%")

    return cl


def train_replay(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config.get('epochs_per_task', 5)
    replay_per_task = config.get('replay_per_task', 200)

    model = SingleHeadMLP().to(device)
    rb = ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)

    for task_id in range(num_tasks):
        tl, vl = benchmark.get_task_dataloaders(task_id)
        opt = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)

        for ep in range(epochs):
            model.train()
            for batch in tl:
                x, y = batch[0].to(device), batch[1].to(device)
                loss = F.cross_entropy(model(x), y)
                if rb.size > 0:
                    rd = rb.sample(256)
                    if rd is not None:
                        rx, ry = rd[0].to(device), rd[1].to(device)
                        loss = loss + F.cross_entropy(model(rx), ry)
                opt.zero_grad(set_to_none=True)
                loss.backward()
                opt.step()

        rb.add_task_samples(benchmark.sample_for_replay(task_id, replay_per_task))
        accs = eval_all(model, benchmark, task_id + 1, device)
        cl.update(task_id, accs)

        if (task_id + 1) % max(1, num_tasks // 10) == 0 or task_id == num_tasks - 1:
            print(f"  Task {task_id}/{num_tasks-1}: Avg={cl.average_accuracy(task_id)*100:.1f}%")

    return cl


def train_packnet(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs = config.get('epochs_per_task', 5)
    retrain_epochs = config.get('retrain_epochs', 2)
    prune_ratio = config.get('prune_ratio', 0.75)

    model = SingleHeadMLP().to(device)
    cl = CLMetrics(num_tasks)

    # Track masks for linear layers in model.net
    masks = {}
    for name, param in model.named_parameters():
        if param.dim() == 2:
            masks[name] = torch.zeros_like(param, dtype=torch.int)

    task_idx = 0

    for task_id in range(num_tasks):
        task_idx += 1
        # Assign free weights
        for n in masks:
            masks[n][masks[n].eq(0)] = task_idx

        # Snapshot frozen
        frozen = {}
        for n in masks:
            fm = (masks[n] > 0) & (masks[n] < task_idx)
            if fm.any():
                frozen[n] = (fm, dict(model.named_parameters())[n].data[fm].clone())

        tl, vl = benchmark.get_task_dataloaders(task_id)

        def do_train(eps, lr):
            opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
            for ep in range(eps):
                model.train()
                for batch in tl:
                    x, y = batch[0].to(device), batch[1].to(device)
                    loss = F.cross_entropy(model(x), y)
                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    for n, p in model.named_parameters():
                        if p.grad is not None and n in masks:
                            p.grad.data[masks[n].ne(task_idx)] = 0
                    opt.step()
                    with torch.no_grad():
                        for n, (fm, fv) in frozen.items():
                            dict(model.named_parameters())[n].data[fm] = fv
                        for n in masks:
                            dict(model.named_parameters())[n].data[masks[n].eq(0)] = 0

        do_train(epochs, 1e-3)

        # Prune
        for n in masks:
            p = dict(model.named_parameters())[n]
            cm = masks[n].eq(task_idx)
            cw = p.data[cm].abs()
            if cw.numel() == 0: continue
            np_ = int(cw.numel() * prune_ratio)
            if 0 < np_ < cw.numel():
                th = cw.kthvalue(np_).values.item()
                pm = cm & (p.data.abs() <= th)
                masks[n][pm] = 0
                p.data[pm] = 0

        # Update frozen after prune
        frozen = {}
        for n in masks:
            fm = (masks[n] > 0) & (masks[n] < task_idx)
            if fm.any():
                frozen[n] = (fm, dict(model.named_parameters())[n].data[fm].clone())

        do_train(retrain_epochs, 1e-4)

        # Eval with masks
        accs = []
        for t in range(task_id + 1):
            tl_t = t + 1
            bk = {}
            with torch.no_grad():
                for n in masks:
                    p = dict(model.named_parameters())[n]
                    zm = ~((masks[n] >= 1) & (masks[n] <= tl_t))
                    if zm.any():
                        bk[n] = (zm, p.data[zm].clone())
                        p.data[zm] = 0
            accs.append(eval_task(model, benchmark.get_task_dataloaders(t)[1], device))
            with torch.no_grad():
                for n, (zm, v) in bk.items():
                    dict(model.named_parameters())[n].data[zm] = v

        cl.update(task_id, accs)

        if (task_id + 1) % max(1, num_tasks // 10) == 0 or task_id == num_tasks - 1:
            cap = sum((masks[n] > 0).sum().item() for n in masks) / sum(masks[n].numel() for n in masks)
            print(f"  Task {task_id}/{num_tasks-1}: Avg={cl.average_accuracy(task_id)*100:.1f}%, Cap={cap*100:.1f}%")

    return cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, required=True,
                        choices=['soft_csc', 'replay', 'packnet'])
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=5)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    device = 'cuda'

    benchmark = PermutedMNIST(num_tasks=args.num_tasks, batch_size=256,
                               num_workers=4, seed=args.seed)
    config = {'num_tasks': args.num_tasks, 'epochs_per_task': args.epochs_per_task,
              'replay_per_task': args.replay_per_task, 'gamma': 0.001, 'beta': 1.0,
              'prune_ratio': 0.75, 'retrain_epochs': 2}

    print(f"Permuted MNIST (single-head): {args.method}, {args.num_tasks} tasks")

    if args.method == 'soft_csc':
        cl = train_soft_csc(benchmark, config, device)
    elif args.method == 'replay':
        cl = train_replay(benchmark, config, device)
    elif args.method == 'packnet':
        cl = train_packnet(benchmark, config, device)

    avg = cl.average_accuracy(args.num_tasks - 1)
    bwt = cl.backward_transfer(args.num_tasks - 1)
    print(f"\nFinal: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl.accuracy_matrix,
        'avg_accuracy': avg, 'bwt': bwt,
    }, f'checkpoints/pmnist_{args.method}_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
