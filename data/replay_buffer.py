"""Experience replay buffer for continual learning.

Simple fixed-size buffer per task. Stores raw tensors for efficiency.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np


class ReplayBuffer:
    """Stores examples from previous tasks for experience replay."""

    def __init__(self, max_per_task=200):
        self.max_per_task = max_per_task
        self.data = []  # list of (x_tensor, y, task_id)
        self.task_counts = {}

    def add_task_samples(self, samples):
        """Add samples from a completed task.

        Args:
            samples: list of (x_tensor, y_int, task_id_int) tuples
        """
        self.data.extend(samples)
        if samples:
            task_id = samples[0][2]
            self.task_counts[task_id] = len(samples)

    @property
    def size(self):
        return len(self.data)

    def sample(self, batch_size):
        """Sample a batch uniformly from all stored examples.

        Returns:
            (x_batch, y_batch, task_id_batch) tensors
        """
        if self.size == 0:
            return None

        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        xs, ys, tids = [], [], []
        for i in indices:
            x, y, tid = self.data[i]
            xs.append(x)
            ys.append(y)
            tids.append(tid)

        return torch.stack(xs), torch.tensor(ys, dtype=torch.long), torch.tensor(tids, dtype=torch.long)

    def get_task_loader(self, task_id, batch_size=64):
        """Get a DataLoader for a specific task's replay data."""
        task_data = [(x, y, t) for x, y, t in self.data if t == task_id]
        if not task_data:
            return None
        ds = ReplayDataset(task_data)
        return DataLoader(ds, batch_size=batch_size, shuffle=True)

    def __len__(self):
        return self.size


class ReplayDataset(Dataset):
    def __init__(self, data):
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]
