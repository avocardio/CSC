"""Permuted MNIST for continual learning scalability experiments."""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np


class PermutedMNISTTask(Dataset):
    """MNIST with a fixed random permutation applied to pixels."""

    def __init__(self, mnist_dataset, permutation, task_id):
        self.dataset = mnist_dataset
        self.permutation = permutation
        self.task_id = task_id

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        x, y = self.dataset[idx]
        # x is (1, 28, 28), flatten, permute, reshape
        flat = x.view(-1)[self.permutation]
        # Reshape to (1, 28, 28) for conv or keep flat for MLP
        x_perm = flat.view(1, 28, 28)
        return x_perm, y, self.task_id


class PermutedMNIST:
    """Generate a sequence of permuted MNIST tasks."""

    def __init__(self, data_root='/mnt/e/datasets/mnist', num_tasks=10,
                 batch_size=128, num_workers=4, seed=42):
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers

        transform = T.Compose([T.ToTensor(), T.Normalize((0.1307,), (0.3081,))])

        self.train_dataset = torchvision.datasets.MNIST(
            root=data_root, train=True, download=False, transform=transform)
        self.test_dataset = torchvision.datasets.MNIST(
            root=data_root, train=False, download=False, transform=transform)

        # Generate permutations
        rng = np.random.RandomState(seed)
        self.permutations = []
        for t in range(num_tasks):
            if t == 0:
                # First task: identity permutation
                self.permutations.append(torch.arange(784))
            else:
                perm = torch.from_numpy(rng.permutation(784).copy())
                self.permutations.append(perm)

    def get_task_dataloaders(self, task_id):
        perm = self.permutations[task_id]
        train_ds = PermutedMNISTTask(self.train_dataset, perm, task_id)
        test_ds = PermutedMNISTTask(self.test_dataset, perm, task_id)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        return train_loader, test_loader

    def sample_for_replay(self, task_id, n_samples=200):
        perm = self.permutations[task_id]
        indices = np.random.choice(len(self.train_dataset), n_samples, replace=False)
        samples = []
        for i in indices:
            x, y = self.train_dataset[i]
            x_perm = x.view(-1)[perm].view(1, 28, 28)
            samples.append((x_perm, y, task_id))
        return samples
