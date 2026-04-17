"""DER++ replay buffer: stores inputs, labels, AND model logits."""

import torch
import numpy as np


class DERBuffer:
    """DER++ buffer: stores (x, y, logits, task_id) for dark experience replay."""

    def __init__(self, max_per_task=200):
        self.max_per_task = max_per_task
        self.data = []  # list of (x_tensor, y_int, logits_tensor, task_id_int)
        self.task_counts = {}

    def add_task_samples(self, samples_with_logits):
        """Add samples with logits from a completed task.

        Args:
            samples_with_logits: list of (x_tensor, y_int, logits_tensor, task_id_int)
        """
        self.data.extend(samples_with_logits)
        if samples_with_logits:
            task_id = samples_with_logits[0][3]
            self.task_counts[task_id] = len(samples_with_logits)

    @property
    def size(self):
        return len(self.data)

    def sample(self, batch_size):
        """Sample a batch uniformly from all stored examples.

        Returns:
            (x_batch, y_batch, logits_batch, task_id_batch) tensors
        """
        if self.size == 0:
            return None

        indices = np.random.choice(self.size, min(batch_size, self.size), replace=False)
        xs, ys, logits, tids = [], [], [], []
        for i in indices:
            x, y, l, tid = self.data[i]
            xs.append(x)
            ys.append(y)
            logits.append(l)
            tids.append(tid)

        return (torch.stack(xs), torch.tensor(ys, dtype=torch.long),
                torch.stack(logits), torch.tensor(tids, dtype=torch.long))
