"""Continual learning evaluation metrics."""

import torch
import numpy as np


@torch.no_grad()
def evaluate_task(model, test_loader, task_id, device='cuda'):
    """Evaluate model accuracy on a single task."""
    model.eval()
    correct = 0
    total = 0
    for batch in test_loader:
        x, y = batch[0].to(device), batch[1].to(device)
        logits = model(x, task_id=task_id)
        preds = logits.argmax(dim=1)
        correct += (preds == y).sum().item()
        total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def evaluate_all_tasks(model, benchmark, num_tasks_seen, device='cuda'):
    """Evaluate on all tasks seen so far. Returns accuracy list."""
    accs = []
    for t in range(num_tasks_seen):
        test_loader = benchmark.get_task_test_loader(t)
        acc = evaluate_task(model, test_loader, t, device)
        accs.append(acc)
    return accs


class CLMetrics:
    """Track continual learning metrics across tasks.

    Maintains the accuracy matrix A[i][j] = accuracy on task j after training on task i.
    """

    def __init__(self, num_tasks):
        self.num_tasks = num_tasks
        self.accuracy_matrix = np.zeros((num_tasks, num_tasks))
        self.bits_history = []  # total bits after each task

    def update(self, current_task, accuracies, total_bits=None):
        """Record accuracies after training on current_task."""
        for j, acc in enumerate(accuracies):
            self.accuracy_matrix[current_task][j] = acc
        if total_bits is not None:
            self.bits_history.append(total_bits)

    def average_accuracy(self, after_task=None):
        """Average accuracy after training on task T (default: last task)."""
        if after_task is None:
            after_task = self.num_tasks - 1
        return np.mean(self.accuracy_matrix[after_task, :after_task + 1])

    def backward_transfer(self, after_task=None):
        """Backward transfer: avg degradation of old tasks."""
        if after_task is None:
            after_task = self.num_tasks - 1
        if after_task == 0:
            return 0.0
        bwt = 0.0
        for j in range(after_task):
            bwt += self.accuracy_matrix[after_task][j] - self.accuracy_matrix[j][j]
        return bwt / after_task

    def forward_transfer(self, after_task=None):
        """Forward transfer: performance on new task before training."""
        if after_task is None:
            after_task = self.num_tasks - 1
        if after_task == 0:
            return 0.0
        fwt = 0.0
        count = 0
        for j in range(1, after_task + 1):
            if j - 1 >= 0:
                fwt += self.accuracy_matrix[j - 1][j]  # acc on task j before training on j
                count += 1
        return fwt / count if count > 0 else 0.0

    def summary(self, after_task=None):
        if after_task is None:
            after_task = self.num_tasks - 1
        return {
            'avg_accuracy': self.average_accuracy(after_task),
            'backward_transfer': self.backward_transfer(after_task),
            'forward_transfer': self.forward_transfer(after_task),
            'bits_history': self.bits_history,
        }

    def print_matrix(self, after_task=None):
        if after_task is None:
            after_task = self.num_tasks - 1
        print("\nAccuracy Matrix (A[i][j] = acc on task j after training on task i):")
        header = "     " + "".join([f"T{j:<6d}" for j in range(after_task + 1)])
        print(header)
        for i in range(after_task + 1):
            row = f"T{i:<3d} "
            for j in range(after_task + 1):
                if j <= i:
                    row += f"{self.accuracy_matrix[i][j]*100:5.1f} "
                else:
                    row += "  -   "
            print(row)
        print(f"\nAvg Accuracy: {self.average_accuracy(after_task)*100:.2f}%")
        print(f"Backward Transfer: {self.backward_transfer(after_task)*100:.2f}%")
