"""Continual-learning metrics: per-task evaluation and the standard
accuracy/BWT/forgetting bookkeeping.
"""
from __future__ import annotations
import torch


@torch.no_grad()
def evaluate_task(model, loader, task_id: int, device: str = 'cuda') -> float:
    model.eval()
    correct, total = 0, 0
    for batch in loader:
        x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
        logits = model(x, task_id=task_id)
        pred = logits.argmax(dim=-1)
        correct += (pred == y).sum().item()
        total += y.numel()
    return correct / max(total, 1)


@torch.no_grad()
def evaluate_all_tasks(model, benchmark, n_seen_tasks: int, device: str = 'cuda') -> list[float]:
    accs = []
    for t in range(n_seen_tasks):
        _, test_loader = benchmark.get_task_dataloaders(t)
        accs.append(evaluate_task(model, test_loader, t, device))
    return accs


class CLMetrics:
    """Holds the accuracy_matrix A[i][j] = accuracy on task j after training on task i.

    Average accuracy = mean over j of A[T-1][j].
    Backward transfer  = mean over j of (A[T-1][j] - A[j][j]) for j < T-1.
    """

    def __init__(self, num_tasks: int):
        self.num_tasks = num_tasks
        self.accuracy_matrix = [[0.0] * num_tasks for _ in range(num_tasks)]

    def update(self, task_idx: int, accs: list[float]):
        for j, a in enumerate(accs):
            self.accuracy_matrix[task_idx][j] = float(a)

    def average_accuracy(self, task_idx: int) -> float:
        row = self.accuracy_matrix[task_idx][: task_idx + 1]
        return sum(row) / max(len(row), 1)

    def final_average(self) -> float:
        return self.average_accuracy(self.num_tasks - 1)

    def backward_transfer(self) -> float:
        T = self.num_tasks
        if T < 2:
            return 0.0
        diffs = [self.accuracy_matrix[T - 1][j] - self.accuracy_matrix[j][j]
                 for j in range(T - 1)]
        return sum(diffs) / max(len(diffs), 1)

    def forgetting(self) -> float:
        # Max forgetting = max over j (max_i<T A[i][j] - A[T-1][j])
        T = self.num_tasks
        if T < 2:
            return 0.0
        forgets = []
        for j in range(T - 1):
            max_prev = max(self.accuracy_matrix[i][j] for i in range(j, T - 1))
            forgets.append(max_prev - self.accuracy_matrix[T - 1][j])
        return sum(forgets) / max(len(forgets), 1)

    def summary(self) -> dict:
        return {
            'final_avg': self.final_average(),
            'backward_transfer': self.backward_transfer(),
            'forgetting': self.forgetting(),
            'matrix': self.accuracy_matrix,
        }

    def print_matrix(self):
        print('\nAccuracy matrix (rows = task trained on, cols = eval task):')
        T = self.num_tasks
        header = '       ' + ' '.join(f'T{j:>2d} ' for j in range(T))
        print(header)
        for i in range(T):
            row = self.accuracy_matrix[i]
            row_str = ' '.join(f'{a*100:5.1f}' for a in row[: i + 1])
            print(f'  T{i:>2d}: {row_str}')
        print(f'\n  final_avg = {self.final_average()*100:.2f}%')
        print(f'  BWT       = {self.backward_transfer()*100:+.2f}%')
        print(f'  forgetting= {self.forgetting()*100:.2f}%')
