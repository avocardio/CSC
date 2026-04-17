"""Split TinyImageNet into sequential tasks for continual learning.

200 classes, 64x64 images. Split into N tasks of 200/N classes.
Standard: 10 tasks x 20 classes.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
import os


def get_tinyimagenet_transforms(train=True):
    if train:
        return T.Compose([
            T.RandomCrop(64, padding=8),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])
    else:
        return T.Compose([
            T.ToTensor(),
            T.Normalize((0.4802, 0.4481, 0.3975), (0.2770, 0.2691, 0.2821)),
        ])


class TaskDatasetTIN(Dataset):
    def __init__(self, dataset, indices, class_mapping, task_id):
        self.dataset = dataset
        self.indices = indices
        self.class_mapping = class_mapping
        self.task_id = task_id

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        x, y = self.dataset[self.indices[idx]]
        return x, self.class_mapping[y], self.task_id


class SplitTinyImageNet:
    def __init__(self, data_root='/mnt/e/datasets/tinyimagenet/tiny-imagenet-200',
                 num_tasks=10, batch_size=128, num_workers=8, seed=42):
        self.num_tasks = num_tasks
        self.classes_per_task = 200 // num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers

        rng = np.random.RandomState(seed)
        self.class_order = rng.permutation(200).tolist()

        train_dir = os.path.join(data_root, 'train')
        val_dir = data_root + '/val'

        self.train_dataset = torchvision.datasets.ImageFolder(
            train_dir, transform=get_tinyimagenet_transforms(train=True))
        self.test_dataset = torchvision.datasets.ImageFolder(
            val_dir, transform=get_tinyimagenet_transforms(train=False))

        # Raw version for replay
        self.train_dataset_raw = torchvision.datasets.ImageFolder(
            train_dir, transform=get_tinyimagenet_transforms(train=False))

        self.train_targets = np.array([s[1] for s in self.train_dataset.samples])
        self.test_targets = np.array([s[1] for s in self.test_dataset.samples])

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
        train_ds = TaskDatasetTIN(self.train_dataset, task['train_indices'],
                                  task['class_mapping'], task_id)
        test_ds = TaskDatasetTIN(self.test_dataset, task['test_indices'],
                                 task['class_mapping'], task_id)
        train_loader = DataLoader(train_ds, batch_size=self.batch_size, shuffle=True,
                                  num_workers=self.num_workers, pin_memory=True,
                                  drop_last=True, persistent_workers=self.num_workers > 0)
        test_loader = DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                                 num_workers=self.num_workers, pin_memory=True,
                                 persistent_workers=self.num_workers > 0)
        return train_loader, test_loader

    def get_task_test_loader(self, task_id):
        task = self.tasks[task_id]
        test_ds = TaskDatasetTIN(self.test_dataset, task['test_indices'],
                                 task['class_mapping'], task_id)
        return DataLoader(test_ds, batch_size=self.batch_size, shuffle=False,
                          num_workers=self.num_workers, pin_memory=True,
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
