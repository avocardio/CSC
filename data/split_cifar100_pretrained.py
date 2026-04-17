"""Split CIFAR-100 with 224x224 resize for pretrained ImageNet models."""

import torchvision
import torchvision.transforms as T
import numpy as np
from torch.utils.data import DataLoader
from data.split_cifar100 import TaskDataset


def get_cifar100_pretrained_transforms(train=True):
    """CIFAR-100 transforms resized to 224x224 for pretrained ResNet."""
    if train:
        return T.Compose([
            T.Resize(224),
            T.RandomCrop(224, padding=28),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return T.Compose([
            T.Resize(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


class SplitCIFAR100Pretrained:
    """Split CIFAR-100 resized to 224x224 for pretrained models."""

    def __init__(self, data_root='/mnt/e/datasets/cifar100', num_tasks=10,
                 batch_size=64, num_workers=8, seed=42):
        self.num_tasks = num_tasks
        self.classes_per_task = 100 // num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers

        rng = np.random.RandomState(seed)
        self.class_order = rng.permutation(100).tolist()

        self.train_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=False,
            transform=get_cifar100_pretrained_transforms(train=True))
        self.test_dataset = torchvision.datasets.CIFAR100(
            root=data_root, train=False, download=False,
            transform=get_cifar100_pretrained_transforms(train=False))
        self.train_dataset_raw = torchvision.datasets.CIFAR100(
            root=data_root, train=True, download=False,
            transform=get_cifar100_pretrained_transforms(train=False))

        self.train_targets = np.array(self.train_dataset.targets)
        self.test_targets = np.array(self.test_dataset.targets)

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
        task = self.tasks[task_id]
        train_ds = TaskDataset(self.train_dataset, task['train_indices'],
                               task['class_mapping'], task_id)
        test_ds = TaskDataset(self.test_dataset, task['test_indices'],
                              task['class_mapping'], task_id)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size,
                                  shuffle=True, num_workers=self.num_workers,
                                  pin_memory=True, drop_last=True,
                                  persistent_workers=self.num_workers > 0)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size,
                                 shuffle=False, num_workers=self.num_workers,
                                 pin_memory=True,
                                 persistent_workers=self.num_workers > 0)
        return train_loader, test_loader

    def get_task_test_loader(self, task_id):
        task = self.tasks[task_id]
        test_ds = TaskDataset(self.test_dataset, task['test_indices'],
                              task['class_mapping'], task_id)
        return DataLoader(test_ds, batch_size=self.batch_size,
                          shuffle=False, num_workers=self.num_workers,
                          pin_memory=True,
                          persistent_workers=self.num_workers > 0)

    def sample_for_replay(self, task_id, n_samples=200):
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
