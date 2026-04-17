"""Split ImageNet-R for continual learning.

ImageNet-R has 200 classes of renditions (art, cartoons, etc.) of ImageNet classes.
Split into 10 tasks of 20 classes each.
Images are 224x224 (standard ImageNet size) — works directly with pretrained ResNet.
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as T
import numpy as np
import os


def get_imagenet_r_transforms(train=True):
    """Standard ImageNet transforms for pretrained models."""
    if train:
        return T.Compose([
            T.RandomResizedCrop(224),
            T.RandomHorizontalFlip(),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
    else:
        return T.Compose([
            T.Resize(256),
            T.CenterCrop(224),
            T.ToTensor(),
            T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])


class TaskDatasetINR(Dataset):
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


class SplitImageNetR:
    """Split ImageNet-R into sequential tasks."""

    def __init__(self, data_root='/mnt/e/datasets/imagenet-r/imagenet-r',
                 num_tasks=10, batch_size=64, num_workers=8, seed=42,
                 val_fraction=0.2):
        self.num_tasks = num_tasks
        self.batch_size = batch_size
        self.num_workers = num_workers

        # Load full dataset
        full_dataset = torchvision.datasets.ImageFolder(
            data_root, transform=get_imagenet_r_transforms(train=True))

        # Also create a non-augmented version for test/replay
        full_dataset_test = torchvision.datasets.ImageFolder(
            data_root, transform=get_imagenet_r_transforms(train=False))

        # Non-augmented for replay storage
        self.train_dataset_raw = torchvision.datasets.ImageFolder(
            data_root, transform=get_imagenet_r_transforms(train=False))

        num_classes = len(full_dataset.classes)
        self.classes_per_task = num_classes // num_tasks
        print(f"ImageNet-R: {num_classes} classes, {len(full_dataset)} images, "
              f"{num_tasks} tasks x {self.classes_per_task} classes")

        # Random class order
        rng = np.random.RandomState(seed)
        self.class_order = rng.permutation(num_classes).tolist()

        # Get all targets
        all_targets = np.array([s[1] for s in full_dataset.samples])

        # Split each class into train/val
        train_indices_all = []
        val_indices_all = []
        for c in range(num_classes):
            class_indices = np.where(all_targets == c)[0]
            rng2 = np.random.RandomState(seed + c)
            rng2.shuffle(class_indices)
            n_val = max(1, int(len(class_indices) * val_fraction))
            val_indices_all.extend(class_indices[:n_val].tolist())
            train_indices_all.extend(class_indices[n_val:].tolist())

        train_set = set(train_indices_all)
        val_set = set(val_indices_all)

        # Store datasets
        self.train_dataset = full_dataset
        self.test_dataset = full_dataset_test

        # Build task splits
        self.tasks = []
        for t in range(num_tasks):
            start = t * self.classes_per_task
            end = start + self.classes_per_task
            task_classes = self.class_order[start:end]
            class_mapping = {c: i for i, c in enumerate(task_classes)}

            task_train = [i for i in train_indices_all if all_targets[i] in task_classes]
            task_val = [i for i in val_indices_all if all_targets[i] in task_classes]

            self.tasks.append({
                'classes': task_classes,
                'class_mapping': class_mapping,
                'train_indices': task_train,
                'test_indices': task_val,
            })

    def get_task_dataloaders(self, task_id):
        task = self.tasks[task_id]
        train_ds = TaskDatasetINR(self.train_dataset, task['train_indices'],
                                  task['class_mapping'], task_id)
        test_ds = TaskDatasetINR(self.test_dataset, task['test_indices'],
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
        test_ds = TaskDatasetINR(self.test_dataset, task['test_indices'],
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
