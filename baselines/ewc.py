"""Elastic Weight Consolidation (EWC) baseline.

Kirkpatrick et al., 2017. Regularization-based continual learning.
Adds a penalty on changing parameters that were important for previous tasks,
measured by the diagonal of the Fisher information matrix.
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from baselines.finetune import SimpleResNet18


def compute_fisher(model, data_loader, task_id, device, n_samples=500):
    """Compute diagonal Fisher information matrix for a task."""
    model.eval()
    fisher = {n: torch.zeros_like(p) for n, p in model.named_parameters() if p.requires_grad}
    count = 0

    for batch in data_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        model.zero_grad()
        logits = model(x, task_id=task_id)
        loss = F.cross_entropy(logits, y)
        loss.backward()

        for n, p in model.named_parameters():
            if p.requires_grad and p.grad is not None:
                fisher[n] += p.grad.data ** 2 * x.size(0)

        count += x.size(0)
        if count >= n_samples:
            break

    for n in fisher:
        fisher[n] /= count

    return fisher


def ewc_loss(model, fisher_dict, params_dict, ewc_lambda):
    """Compute EWC regularization loss."""
    loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for n, p in model.named_parameters():
        if p.requires_grad and n in fisher_dict:
            loss += (fisher_dict[n] * (p - params_dict[n]) ** 2).sum()
    return 0.5 * ewc_lambda * loss


def train_ewc(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs_per_task = config.get('epochs_per_task', 50)
    lr = config.get('lr', 1e-3)
    ewc_lambda = config.get('ewc_lambda', 1000)
    classes_per_task = 100 // num_tasks

    model = SimpleResNet18(classes_per_task, num_tasks).to(device)
    cl_metrics = CLMetrics(num_tasks)

    # Store Fisher and params for each task
    all_fishers = []
    all_params = []

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

                with torch.amp.autocast('cuda'):
                    logits = model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                # EWC penalty (outside autocast for stability)
                for fish, params in zip(all_fishers, all_params):
                    loss = loss + ewc_loss(model, fish, params, ewc_lambda)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 10 == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

        # Compute and store Fisher
        fisher = compute_fisher(model, train_loader, task_id, device)
        params = {n: p.data.clone() for n, p in model.named_parameters() if p.requires_grad}
        all_fishers.append(fisher)
        all_params.append(params)

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
    parser.add_argument('--ewc_lambda', type=float, default=1000)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size, seed=args.seed)
    config = vars(args)
    cl_metrics = train_ewc(benchmark, config)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
    }, f'checkpoints/ewc_t{args.num_tasks}_l{args.ewc_lambda}.pt')


if __name__ == '__main__':
    main()
