"""Synaptic Intelligence (SI) baseline — Zenke et al., ICML 2017.

Computes per-parameter importance online during training based on each
parameter's contribution to loss reduction. Adds a quadratic penalty
discouraging changes to important parameters.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from baselines.finetune import SimpleResNet18


class SynapticIntelligence:
    """Track per-parameter importance using path integral of gradient * delta."""

    def __init__(self, model, device='cuda', damping=0.1):
        self.model = model
        self.device = device
        self.damping = damping  # xi in original paper

        # Importance accumulator (omega in paper)
        self.omega = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                      if p.requires_grad}
        # Running sum of gradient * parameter change
        self.running_sum = {n: torch.zeros_like(p) for n, p in model.named_parameters()
                            if p.requires_grad}
        # Parameter snapshot at task start
        self.prev_params = {n: p.data.clone() for n, p in model.named_parameters()
                            if p.requires_grad}
        # Parameter snapshot at previous step (for delta computation)
        self.step_params = {n: p.data.clone() for n, p in model.named_parameters()
                            if p.requires_grad}

    def update_running_sum(self):
        """Call after each optimizer step to accumulate importance."""
        for n, p in self.model.named_parameters():
            if n in self.running_sum and p.grad is not None:
                delta = p.data - self.step_params[n]
                self.running_sum[n] += (-p.grad.data * delta)
                self.step_params[n] = p.data.clone()

    def update_omega(self):
        """Call at end of each task to consolidate importance."""
        for n, p in self.model.named_parameters():
            if n in self.omega:
                delta = p.data - self.prev_params[n]
                self.omega[n] += self.running_sum[n] / (delta ** 2 + self.damping)
                # Reset for next task
                self.running_sum[n].zero_()
                self.prev_params[n] = p.data.clone()
                self.step_params[n] = p.data.clone()

    def penalty(self):
        """Compute SI regularization loss."""
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.model.named_parameters():
            if n in self.omega:
                loss += (self.omega[n] * (p - self.prev_params[n]) ** 2).sum()
        return loss


def train_si(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs_per_task = config.get('epochs_per_task', 50)
    lr = config.get('lr', 1e-3)
    si_lambda = config.get('si_lambda', 1.0)
    classes_per_task = len(benchmark.tasks[0]['classes'])

    model = SimpleResNet18(classes_per_task, num_tasks).to(device)
    cl_metrics = CLMetrics(num_tasks)
    si = SynapticIntelligence(model, device)

    for task_id in range(num_tasks):
        print(f"\n{'='*40} TASK {task_id} {'='*40}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs_per_task * len(train_loader))
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(epochs_per_task):
            model.train()
            for batch in train_loader:
                x, y = batch[0].to(device), batch[1].to(device)

                logits = model(x, task_id=task_id)
                loss = F.cross_entropy(logits, y)

                # SI penalty
                if task_id > 0:
                    loss = loss + si_lambda * si.penalty()

                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()

                # Update SI running sum after each step
                si.update_running_sum()

            if (epoch + 1) % 10 == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # Consolidate importance at task boundary
        si.update_omega()

        all_accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl_metrics.update(task_id, all_accs)
        for j, a in enumerate(all_accs):
            print(f"  Task {j}: {a*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")

    cl_metrics.print_matrix()
    return cl_metrics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--si_lambda', type=float, default=1.0)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)
    cl_metrics = train_si(benchmark, config)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
    }, f'checkpoints/si_t{args.num_tasks}_l{args.si_lambda}.pt')


if __name__ == '__main__':
    main()
