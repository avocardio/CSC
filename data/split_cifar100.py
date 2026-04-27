"""Split CIFAR-100 into sequential tasks for continual learning.

Standard benchmark: 100 classes split into N tasks of (100/N) classes each.
Supports 10 tasks x 10 classes or 20 tasks x 5 classes.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Subset
import torchvision
import torchvision.transforms as T
import numpy as np


def get_cifar100_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
            T.RandomErasing(p=0.5),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
        ])


def get_cifar10_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomCrop(32, padding=4),
            T.RandomHorizontalFlip(),
            T.AutoAugment(T.AutoAugmentPolicy.CIFAR10),
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
            T.RandomErasing(p=0.5),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4914, 0.4822, 0.4465), (0.2470, 0.2435, 0.2616)),
        ])


class TaskDataset(Dataset):
    """Wraps a subset of a dataset and remaps labels to 0..num_classes-1."""

    def __init__(self, dataset, indices, class_mapping, task_id):
        self.dataset = dataset
        self.indices = indices
        self.class_mapping = class_mapping  # original_label -> local_label
        self.task_id = task_id

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        local_y = self.class_mapping[y]
        return x, local_y, self.task_id


class SplitCIFAR100:
    """Split CIFAR-100 into sequential tasks."""

    def __init__(self, data_root='/mnt/e/datasets/cifar100', num_tasks=10,
                 batch_size=128, num_workers=4, seed=42):
        self.num_tasks = num_tasks
        self.classes_per_task = 100 // num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Fix class ordering with seed
        rng = np.random.RandomState(seed)
        self.class_order = rng.permutation(100).tolist()

        # Load full datasets
        self.train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=get_cifar100_transforms(train=True))
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=True,
            transform=get_cifar100_transforms(train=False))

        # Also load raw (no augmentation) for replay buffer storage
        self.train_dataset_raw = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=True,
            transform=get_cifar100_transforms(train=False))

        # Get targets as numpy array for fast indexing
        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)

        # Precompute task splits
        self.tasks = []
        for t in range(num_tasks):
            start = t * self.classes_per_task
            end = start + self.classes_per_task
            task_classes = self.class_order[start:end]
            class_mapping = {c: i for i, c in enumerate(task_classes)}

            train_indices = np.where(np.isin(self.train_targets, task_classes))[0].tolist()
            test_indices = np.where(np.isin(self.test_targets, task_classes))[0].tolist()

            self.tasks.append({
                'classes': task_classes,
                'class_mapping': class_mapping,
                'train_indices': train_indices,
                'test_indices': test_indices,
            })

    def get_task_dataloaders(self, task_id):
        """Get train and test dataloaders for a specific task."""
        task = self.tasks[task_id]

        train_ds = TaskDataset(self.train_dataset, task['train_indices'],
                               task['class_mapping'], task_id)
        test_ds = TaskDataset(self.test_dataset, task['test_indices'],
                              task['class_mapping'], task_id)

        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True, drop_last=True,
                                  persistent_workers=self.num_workers > 0,
                                  prefetch_factor=3 if self.num_workers > 0 else None)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True,
                                 persistent_workers=self.num_workers > 0,
                                 prefetch_factor=3 if self.num_workers > 0 else None)

        return train_loader, test_loader

    def get_task_test_loader(self, task_id):
        """Get just the test loader for a task."""
        task = self.tasks[task_id]
        test_ds = TaskDataset(self.test_dataset, task['test_indices'],
                              task['class_mapping'], task_id)
        return DataLoader(test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0,
                          prefetch_factor=3 if self.num_workers > 0 else None)

    def sample_for_replay(self, task_id, n_samples=200):
        """Sample examples from a task for the replay buffer."""
        task = self.tasks[task_id]
        indices = task['train_indices']
        rng = np.random.RandomState(task_id)
        selected = rng.choice(len(indices), min(n_samples, len(indices)), replace=False)
        samples = []
        for i in selected:
            x, y = self.train_dataset_raw[indices[i]]
            local_y = task['class_mapping'][y]
            samples.append((x, local_y, task_id))
        return samples


class SplitCIFAR10:
    """Single-task CIFAR-10 for Phase 1 validation."""

    def __init__(self, data_root='/mnt/e/datasets/cifar10', batch_size=128, num_workers=4):
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.train_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=True, download=True,
            transform=get_cifar10_transforms(train=True))
        self.test_dataset = torchvision.datasets.CIFAR10(
            root=data_root, train=False, download=True,
            transform=get_cifar10_transforms(train=False))

    def get_dataloaders(self):
        train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True, drop_last=True)
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True)
        return train_loader, test_loader
