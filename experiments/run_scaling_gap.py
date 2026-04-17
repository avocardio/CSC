"""Scaling gap experiment: random vs learned importance across task counts.

Methods:
A) Replay-only (uniform LR)
B) Random fixed scaling (random per-channel factors, fixed forever)
C) Importance-only (learned bit-depths via compression, no quantization)
D) Re-randomized scaling (random factors re-shuffled at each task boundary)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import numpy as np
from training.metrics import CLMetrics, evaluate_task, evaluate_all_tasks
from data.split_cifar100 import SplitCIFAR100
from data.replay_buffer import ReplayBuffer
from models.quantization import DifferentiableQuantizer, CompressionGranularity


class ScalingModel(nn.Module):
    """ResNet-18 with optional importance parameters for gradient scaling."""

    def __init__(self, num_classes_per_task, num_tasks, has_importance=False, init_bit_depth=8.0):
        super().__init__()
        import torchvision.models as models
        backbone = models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()
        backbone.fc = nn.Identity()
        self.backbone = backbone

        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])
        for h in self.heads:
            nn.init.normal_(h.weight, 0, 0.01)
            nn.init.constant_(h.bias, 0)

        self.has_importance = has_importance
        self.num_tasks = num_tasks

        # Collect conv layer info for importance
        self._conv_info = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                self._conv_info.append((name, module.out_channels))

        if has_importance:
            self.importance = nn.ParameterDict()
            for name, n_ch in self._conv_info:
                safe = name.replace('.', '_')
                self.importance[safe] = nn.Parameter(
                    torch.full((n_ch,), init_bit_depth))

    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def compute_avg_importance(self):
        """Average of importance params (for compression loss)."""
        if not self.has_importance:
            return torch.tensor(0.0)
        total, count = 0.0, 0
        for safe, bits in self.importance.items():
            total += bits.clamp(min=0).sum()
            count += bits.numel()
        return total / max(count, 1)

    def get_conv_modules(self):
        """Get list of (name, conv_module, safe_name) for gradient scaling."""
        result = []
        for name, module in self.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                safe = name.replace('.', '_')
                result.append((name, module, safe))
        return result


def scale_gradients(model, scaling_factors, beta, task_id):
    """Apply per-channel gradient scaling using given factors."""
    for name, module, safe in model.get_conv_modules():
        if safe in scaling_factors and module.weight.grad is not None:
            factors = scaling_factors[safe]
            scale = 1.0 / (1.0 + beta * factors.view(-1, 1, 1, 1))
            module.weight.grad.data *= scale.expand_as(module.weight.grad)

    # Scale importance param gradients if they exist
    if model.has_importance:
        for safe, bits in model.importance.items():
            if bits.grad is not None and safe in scaling_factors:
                factors = scaling_factors[safe]
                bits.grad.data *= 1.0 / (1.0 + beta * factors)

    # Protect other task heads
    for pname, param in model.named_parameters():
        if 'heads.' in pname and param.grad is not None:
            hi = int(pname.split('heads.')[1].split('.')[0])
            if hi != task_id:
                param.grad.data.fill_(0)


def get_learned_factors(model):
    """Get current bit-depths as scaling factors."""
    factors = {}
    for safe, bits in model.importance.items():
        factors[safe] = bits.clamp(min=0).detach()
    return factors


def get_random_factors(model, seed=0):
    """Generate random fixed scaling factors per channel."""
    rng = torch.Generator()
    rng.manual_seed(seed)
    factors = {}
    for name, module, safe in model.get_conv_modules():
        # Uniform(0.1, 8.0) to match bit-depth range
        factors[safe] = torch.rand(module.out_channels, generator=rng) * 7.9 + 0.1
    return factors


def train_method(method, benchmark, num_tasks, epochs_per_task, replay_per_task,
                 gamma, beta, device, seed=42):
    """Train one method and return CLMetrics."""
    torch.manual_seed(seed); torch.cuda.manual_seed(seed)

    classes_per_task = len(benchmark.tasks[0]['classes'])
    has_importance = method in ('learned', 'rerandom')
    model = ScalingModel(classes_per_task, num_tasks,
                         has_importance=(method == 'learned')).to(device)
    rb = ReplayBuffer(max_per_task=replay_per_task)
    cl = CLMetrics(num_tasks)

    # Initialize scaling factors
    if method == 'random_fixed':
        scaling = {k: v.to(device) for k, v in get_random_factors(model, seed=seed).items()}
    elif method == 'rerandom':
        scaling = {k: v.to(device) for k, v in get_random_factors(model, seed=seed).items()}
    else:
        scaling = None

    scaler = torch.amp.GradScaler('cuda')

    for task_id in range(num_tasks):
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        # Re-randomize at task boundary for method D
        if method == 'rerandom':
            scaling = {k: v.to(device) for k, v in
                       get_random_factors(model, seed=seed + task_id * 1000).items()}

        # Optimizer
        if method == 'learned':
            imp_params = list(model.importance.parameters())
            imp_ids = {id(p) for p in imp_params}
            other = [p for p in model.parameters() if id(p) not in imp_ids]
            opt = torch.optim.AdamW([
                {'params': other, 'lr': 1e-3, 'weight_decay': 5e-4},
                {'params': imp_params, 'lr': 0.5, 'eps': 1e-3, 'weight_decay': 0},
            ])
        else:
            opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=5e-4)

        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=epochs_per_task * len(train_loader))

        for epoch in range(epochs_per_task):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                    # Compression loss (learned only)
                    if method == 'learned':
                        Q = model.compute_avg_importance()
                        loss = loss + gamma * Q

                    # Replay
                    if rb.size > 0:
                        buf = rb.sample(64)
                        if buf:
                            bx, by, bt = buf
                            bx, by, bt = bx.to(device), by.to(device), bt.to(device)
                            for tid in bt.unique():
                                m = bt == tid
                                loss += F.cross_entropy(model(bx[m], task_id=tid.item()), by[m]) / len(bt.unique())

                opt.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(opt)

                # Apply gradient scaling
                if method == 'replay':
                    # Just protect heads
                    for pn, p in model.named_parameters():
                        if 'heads.' in pn and p.grad is not None:
                            hi = int(pn.split('heads.')[1].split('.')[0])
                            if hi != task_id: p.grad.data.fill_(0)
                elif method == 'learned':
                    factors = get_learned_factors(model)
                    scale_gradients(model, factors, beta, task_id)
                else:  # random_fixed or rerandom
                    scale_gradients(model, scaling, beta, task_id)

                scaler.step(opt)
                scaler.update()
                scheduler.step()

        rb.add_task_samples(benchmark.sample_for_replay(task_id, replay_per_task))

        accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl.update(task_id, accs)

        if (task_id + 1) % max(1, num_tasks // 5) == 0 or task_id == num_tasks - 1:
            print(f"  [{method}] Task {task_id}/{num_tasks-1}: "
                  f"Avg={cl.average_accuracy(task_id)*100:.1f}%")

    return cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', type=str, default='cifar100',
                        choices=['cifar100', 'pmnist'])
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--replay_per_task', type=int, default=200)
    parser.add_argument('--gamma', type=float, default=0.001)
    parser.add_argument('--beta', type=float, default=1.0)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--method', type=str, default='all',
                        choices=['all', 'replay', 'random_fixed', 'learned', 'rerandom'])
    args = parser.parse_args()

    device = 'cuda'

    if args.benchmark == 'cifar100':
        benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=128,
                                   num_workers=8, seed=args.seed)
    else:
        # PMNIST handled separately
        print("Use run_scaling_gap_pmnist.py for PMNIST")
        return

    methods = ['replay', 'random_fixed', 'learned', 'rerandom'] if args.method == 'all' else [args.method]
    results = {}

    for method in methods:
        print(f"\n{'='*50}")
        print(f"Method: {method}, {args.num_tasks} tasks")
        print(f"{'='*50}")

        cl = train_method(method, benchmark, args.num_tasks, args.epochs_per_task,
                          args.replay_per_task, args.gamma, args.beta, device, args.seed)
        avg = cl.average_accuracy(args.num_tasks - 1)
        bwt = cl.backward_transfer(args.num_tasks - 1)
        results[method] = {'avg': avg, 'bwt': bwt}
        print(f"  Final: Avg={avg*100:.2f}%, BWT={bwt*100:.2f}%")

    # Summary
    print(f"\n{'='*50}")
    print(f"SUMMARY ({args.benchmark}, {args.num_tasks} tasks)")
    print(f"{'='*50}")
    for method, r in results.items():
        print(f"  {method:15s}: {r['avg']*100:.2f}% (BWT: {r['bwt']*100:.2f}%)")

    os.makedirs('checkpoints', exist_ok=True)
    torch.save(results, f'checkpoints/scaling_gap_{args.benchmark}_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
