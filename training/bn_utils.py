"""Per-task BatchNorm state management for fair CL evaluation."""

import torch
import torch.nn as nn


class PerTaskBNTracker:
    """Saves and restores full BN state (running stats + affine params) per task."""

    def __init__(self):
        self.task_states = {}

    def save(self, model, task_id):
        """Save full BN state after training on task_id."""
        state = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                state[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone(),
                }
        self.task_states[task_id] = state

    def swap(self, model, task_id):
        """Swap in BN state for task_id, return old state for restore."""
        if task_id not in self.task_states:
            return None
        old = {}
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                old[name] = {
                    'running_mean': module.running_mean.clone(),
                    'running_var': module.running_var.clone(),
                    'weight': module.weight.data.clone(),
                    'bias': module.bias.data.clone(),
                }
                if name in self.task_states[task_id]:
                    s = self.task_states[task_id][name]
                    module.running_mean.copy_(s['running_mean'])
                    module.running_var.copy_(s['running_var'])
                    module.weight.data.copy_(s['weight'])
                    module.bias.data.copy_(s['bias'])
        return old

    def restore(self, model, old_state):
        """Restore BN state from saved dict."""
        if old_state is None:
            return
        for name, module in model.named_modules():
            if isinstance(module, nn.BatchNorm2d) and name in old_state:
                s = old_state[name]
                module.running_mean.copy_(s['running_mean'])
                module.running_var.copy_(s['running_var'])
                module.weight.data.copy_(s['weight'])
                module.bias.data.copy_(s['bias'])


def evaluate_task_with_bn(model, test_loader, task_id, bn_tracker, device='cuda'):
    """Evaluate a single task with per-task BN state."""
    old = bn_tracker.swap(model, task_id)
    from training.metrics import evaluate_task
    acc = evaluate_task(model, test_loader, task_id, device)
    bn_tracker.restore(model, old)
    return acc


def evaluate_all_with_bn(model, benchmark, num_tasks, bn_tracker, device='cuda'):
    """Evaluate all tasks with per-task BN state."""
    accs = []
    for t in range(num_tasks):
        test_loader = benchmark.get_task_test_loader(t)
        acc = evaluate_task_with_bn(model, test_loader, t, bn_tracker, device)
        accs.append(acc)
    return accs
