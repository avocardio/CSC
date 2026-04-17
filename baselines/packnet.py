"""PackNet baseline (Mallya & Lazebnik, CVPR 2018).

Matches the original implementation at github.com/arunmallya/packnet.

Key insight from original code:
- Mask values: 0=pruned, t=belongs to task t
- Before training task t: make_finetuning_mask() sets all mask==0 to mask==t
- During training: only mask==t weights get gradients (make_grads_zero)
- After each step: mask==0 weights re-zeroed (make_pruned_zero)
- Prune: some mask==t weights go back to mask==0
- Eval task k: apply_mask(k) zeros mask==0 and mask>k weights
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import copy
from training.metrics import evaluate_task, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from baselines.finetune import SimpleResNet18


class PackNetTrainer:
    def __init__(self, model, prune_ratio=0.75, device='cuda', train_bn=True):
        self.model = model.to(device)
        self.device = device
        self.prune_ratio = prune_ratio
        self.train_bn = train_bn
        self.current_task_idx = 0  # 1-indexed task label

        # Per-weight mask: 0=pruned/free, t=belongs to task t
        self.masks = {}
        for name, module in self.model.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)) and 'heads' not in name:
                self.masks[name] = torch.zeros_like(module.weight.data, dtype=torch.int)

        # Per-task BN running stats
        self.task_bn_stats = {}

    def _get_frozen_mask(self):
        """Get mask of weights owned by previous tasks (not current, not free)."""
        masks = {}
        for name in self.masks:
            masks[name] = (self.masks[name] > 0) & (self.masks[name] < self.current_task_idx)
        return masks

    def make_finetuning_mask(self):
        """Before training new task: assign all free weights (mask==0) to current task."""
        self.current_task_idx += 1
        for name in self.masks:
            self.masks[name][self.masks[name].eq(0)] = self.current_task_idx

    def make_grads_zero(self, current_task_id):
        """Zero gradients on weights not belonging to current task.
        Also protects heads of other tasks."""
        for name, module in self.model.named_modules():
            if name in self.masks:
                if module.weight.grad is not None:
                    module.weight.grad.data[self.masks[name].ne(self.current_task_idx)] = 0
                if hasattr(module, 'bias') and module.bias is not None and module.bias.grad is not None:
                    module.bias.grad.data.fill_(0)  # freeze biases
            elif isinstance(module, nn.BatchNorm2d) and not self.train_bn:
                if module.weight.grad is not None:
                    module.weight.grad.data.fill_(0)
                if module.bias.grad is not None:
                    module.bias.grad.data.fill_(0)

        # Protect heads of other tasks
        for name, param in self.model.named_parameters():
            if 'heads.' in name and param.grad is not None:
                head_idx = int(name.split('heads.')[1].split('.')[0])
                if head_idx != current_task_id:
                    param.grad.data.fill_(0)

    def make_pruned_zero(self):
        """After each step: zero weights where mask==0."""
        for name, module in self.model.named_modules():
            if name in self.masks:
                module.weight.data[self.masks[name].eq(0)] = 0.0

    def prune(self):
        """Prune lowest-magnitude weights of current task per layer."""
        for name, module in self.model.named_modules():
            if name not in self.masks:
                continue
            mask = self.masks[name]
            # Only prune weights belonging to current task
            current_task_mask = mask.eq(self.current_task_idx)
            current_weights = module.weight.data[current_task_mask].abs()

            if current_weights.numel() == 0:
                continue

            n_prune = int(current_weights.numel() * self.prune_ratio)
            if 0 < n_prune < current_weights.numel():
                threshold = current_weights.kthvalue(n_prune).values.item()
                # Pruned weights go back to mask=0
                prune_mask = current_task_mask & (module.weight.data.abs() <= threshold)
                mask[prune_mask] = 0
                module.weight.data[prune_mask] = 0.0

        self._report_capacity()

    def apply_mask(self, task_idx):
        """For evaluation: zero weights not relevant to task_idx.
        Zeros mask==0 (pruned) and mask>task_idx (future tasks)."""
        backups = {}
        for name, module in self.model.named_modules():
            if name in self.masks:
                mask = self.masks[name]
                zero_mask = mask.eq(0) | mask.gt(task_idx)
                if zero_mask.any():
                    backups[name] = (zero_mask.clone(), module.weight.data[zero_mask].clone())
                    module.weight.data[zero_mask] = 0.0
        return backups

    def restore_mask(self, backups):
        """Restore weights after evaluation."""
        for name, module in self.model.named_modules():
            if name in backups:
                zero_mask, values = backups[name]
                module.weight.data[zero_mask] = values

    def _save_bn_stats(self, task_idx):
        """Save ALL BN state: running stats + learned affine params."""
        stats = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                stats[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone(),
                }
        self.task_bn_stats[task_idx] = stats

    def _swap_bn_stats(self, task_idx):
        """Swap in full BN state for a task. Returns old state for restore."""
        old = {}
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                old[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone(),
                }
                if task_idx in self.task_bn_stats and name in self.task_bn_stats[task_idx]:
                    s = self.task_bn_stats[task_idx][name]
                    module.running_mean.copy_(s['running_mean'])
                    module.running_var.copy_(s['running_var'])
                    module.weight.data.copy_(s['weight'])
                    module.bias.data.copy_(s['bias'])
        return old

    def _restore_bn(self, old):
        for name, module in self.model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in old:
                module.running_mean.copy_(old[name]['running_mean'])
                module.running_var.copy_(old[name]['running_var'])
                module.weight.data.copy_(old[name]['weight'])
                module.bias.data.copy_(old[name]['bias'])

    def _report_capacity(self):
        total = sum(self.masks[n].numel() for n in self.masks)
        used = sum((self.masks[n] > 0).sum().item() for n in self.masks)
        print(f"  Capacity: {used}/{total} ({used/total*100:.1f}%) assigned")

    def evaluate_task_k(self, test_loader, task_idx):
        """Evaluate task k with mask + BN swap."""
        backups = self.apply_mask(task_idx)
        old_bn = self._swap_bn_stats(task_idx)
        acc = evaluate_task(self.model, test_loader, task_idx - 1, self.device)  # task_idx is 1-indexed
        self._restore_bn(old_bn)
        self.restore_mask(backups)
        return acc

    def evaluate_all(self, benchmark, num_tasks):
        accs = []
        for t in range(num_tasks):
            test_loader = benchmark.get_task_test_loader(t)
            acc = self.evaluate_task_k(test_loader, t + 1)  # 1-indexed
            accs.append(acc)
        return accs

    def train_task(self, train_loader, test_loader, task_id,
                   train_epochs=50, retrain_epochs=10):
        """Full PackNet procedure for one task."""
        # Step 1: Assign free weights to this task
        self.make_finetuning_mask()
        print(f"  Assigned free weights to task {self.current_task_idx}")

        # Step 2: Train
        print(f"  Training ({train_epochs} epochs)...")
        self._train_loop(train_loader, task_id, train_epochs, lr=1e-3)

        acc_before = evaluate_task(self.model, test_loader, task_id, self.device)
        print(f"  Pre-prune accuracy: {acc_before*100:.1f}%")

        # Step 3: Prune
        print(f"  Pruning (ratio={self.prune_ratio})...")
        self.prune()

        acc_after = evaluate_task(self.model, test_loader, task_id, self.device)
        print(f"  Post-prune accuracy: {acc_after*100:.1f}%")

        # Step 4: Retrain
        print(f"  Retraining ({retrain_epochs} epochs at lr=1e-4)...")
        self._train_loop(train_loader, task_id, retrain_epochs, lr=1e-4)

        acc_retrain = evaluate_task(self.model, test_loader, task_id, self.device)
        print(f"  Post-retrain accuracy: {acc_retrain*100:.1f}%")

        # Step 5: Save BN stats
        self._save_bn_stats(self.current_task_idx)

    def _train_loop(self, train_loader, task_id, epochs, lr):
        # Snapshot frozen weights to force-restore after each step
        frozen_mask = self._get_frozen_mask()
        frozen_vals = {}
        for name in frozen_mask:
            if frozen_mask[name].any():
                module = dict(self.model.named_modules())[name]
                frozen_vals[name] = module.weight.data[frozen_mask[name]].clone()

        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr, weight_decay=5e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs * len(train_loader))
        scaler = torch.amp.GradScaler('cuda')

        for epoch in range(epochs):
            self.model.train()
            for batch in train_loader:
                x, y = batch[0].to(self.device), batch[1].to(self.device)

                with torch.amp.autocast('cuda'):
                    logits = self.model(x, task_id=task_id)
                    loss = F.cross_entropy(logits, y)

                optimizer.zero_grad(set_to_none=True)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                self.make_grads_zero(task_id)
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()

                # Force-restore frozen weights (prevents drift from weight decay/AMP)
                with torch.no_grad():
                    for name, vals in frozen_vals.items():
                        module = dict(self.model.named_modules())[name]
                        module.weight.data[frozen_mask[name]] = vals

                # Re-zero pruned weights
                self.make_pruned_zero()

            if (epoch + 1) % 10 == 0 or epoch == 0:
                acc = evaluate_task(self.model, train_loader, task_id, self.device)
                print(f"    Epoch {epoch+1}: TrainAcc={acc*100:.1f}%")

    def get_capacity_used(self):
        total = sum(self.masks[n].numel() for n in self.masks)
        used = sum((self.masks[n] > 0).sum().item() for n in self.masks)
        return used / total if total > 0 else 0


def train_packnet(benchmark, config, device='cuda'):
    num_tasks = config['num_tasks']
    epochs_per_task = config.get('epochs_per_task', 50)
    retrain_epochs = config.get('retrain_epochs', 10)
    prune_ratio = config.get('prune_ratio', 0.75)
    classes_per_task = 100 // num_tasks

    model = SimpleResNet18(classes_per_task, num_tasks).to(device)
    trainer = PackNetTrainer(model, prune_ratio=prune_ratio, device=device)
    cl_metrics = CLMetrics(num_tasks)

    for task_id in range(num_tasks):
        print(f"\n{'='*40} TASK {task_id} {'='*40}")
        train_loader, test_loader = benchmark.get_task_dataloaders(task_id)

        trainer.train_task(train_loader, test_loader, task_id,
                           train_epochs=epochs_per_task, retrain_epochs=retrain_epochs)

        all_accs = trainer.evaluate_all(benchmark, task_id + 1)
        capacity = trainer.get_capacity_used()
        cl_metrics.update(task_id, all_accs)

        for j, a in enumerate(all_accs):
            print(f"  Task {j}: {a*100:.1f}%")
        print(f"  Avg: {cl_metrics.average_accuracy(task_id)*100:.2f}%")
        print(f"  Capacity used: {capacity*100:.1f}%")

    cl_metrics.print_matrix()
    return cl_metrics, trainer


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=10)
    parser.add_argument('--epochs_per_task', type=int, default=50)
    parser.add_argument('--retrain_epochs', type=int, default=10)
    parser.add_argument('--prune_ratio', type=float, default=0.75)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)

    benchmark = SplitCIFAR100(num_tasks=args.num_tasks, batch_size=args.batch_size,
                               num_workers=8, seed=args.seed)
    config = vars(args)
    cl_metrics, trainer = train_packnet(benchmark, config)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'summary': cl_metrics.summary(),
        'capacity_used': trainer.get_capacity_used(),
    }, f'checkpoints/packnet_t{args.num_tasks}_p{args.prune_ratio}.pt')


if __name__ == '__main__':
    main()
