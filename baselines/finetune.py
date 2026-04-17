"""Naive fine-tuning baseline for continual learning.

No protection against catastrophic forgetting — sequential training
on each task. This is the lower bound baseline.
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


class SimpleResNet18(nn.Module):
    """Standard ResNet-18 without quantization, multi-head."""

    def __init__(self, num_classes_per_task=10, num_tasks=10, image_size=32):
        super().__init__()
        import torchvision.models as models
        self.backbone = models.resnet18(weights=None)
        if image_size <= 32:
            self.backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.backbone.maxpool = nn.Identity()
        self.backbone.fc = nn.Identity()
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])

    def forward(self, x, task_id=None):
        features = self.backbone(x)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)


def train_finetune(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs_per_task = config.get('epochs_per_task', 50)
    lr = config.get('lr', 1e-3)
    batch_size = config.get('batch_size', 128)
    classes_per_task = 100 // num_tasks

    model = SimpleResNet18(classes_per_task, num_tasks).to(device)
    cl_metrics = CLMetrics(num_tasks)

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
                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

            if (epoch + 1) % 10 == 0:
                acc = evaluate_task(model, test_loader, task_id, device)
                print(f"  Epoch {epoch+1}: Acc={acc*100:.1f}%")

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
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size, seed=args.seed)
    config = vars(args)
    cl_metrics = train_finetune(benchmark, config)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
    }, f'checkpoints/finetune_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
