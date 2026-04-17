"""Loss of Plasticity benchmark: Incremental CIFAR-100.

Reimplements the core benchmark from Dohare et al. (Nature, 2024).
Starts with 5 classes, adds 5 every 200 epochs, 4000 epochs total.

Compares:
A) Standard backprop (baseline — loses plasticity)
B) CBP with contribution utility (their method)
C) Compression-guided replacement (our method — use bit-depth as utility)
D) CBP with random utility (control)
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as T
import numpy as np
import argparse
import time
from torch.utils.data import DataLoader, Subset


class SimpleResNet18(nn.Module):
    """ResNet-18 that exposes intermediate features for utility computation."""

    def __init__(self, num_classes=100):
        super().__init__()
        backbone = torchvision.models.resnet18(weights=None)
        backbone.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
        backbone.maxpool = nn.Identity()

        self.layer0 = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4
        self.avgpool = backbone.avgpool
        self.fc = nn.Linear(512, num_classes)

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, return_features=False):
        features = []
        x = self.layer0(x)
        features.append(x)
        x = self.layer1(x)
        features.append(x)
        x = self.layer2(x)
        features.append(x)
        x = self.layer3(x)
        features.append(x)
        x = self.layer4(x)
        features.append(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        out = self.fc(x)
        if return_features:
            return out, features
        return out


class UnitReplacer:
    """Replaces low-utility units. Supports different utility measures."""

    def __init__(self, model, utility_type='contribution', replacement_rate=0.001,
                 decay_rate=0.99, maturity_threshold=100, device='cuda'):
        self.model = model
        self.utility_type = utility_type
        self.replacement_rate = replacement_rate
        self.decay_rate = decay_rate
        self.maturity_threshold = maturity_threshold
        self.device = device

        # Get conv layers for utility tracking
        self.conv_layers = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.out_channels > 3:  # skip first conv
                self.conv_layers.append((name, module))

        # Utility and age per channel
        self.utility = {name: torch.zeros(m.out_channels, device=device)
                        for name, m in self.conv_layers}
        self.ages = {name: torch.zeros(m.out_channels, device=device)
                     for name, m in self.conv_layers}

        # For compression utility: learnable bit-depths
        self.bit_depths = {}
        if utility_type == 'compression':
            for name, m in self.conv_layers:
                self.bit_depths[name] = nn.Parameter(
                    torch.full((m.out_channels,), 8.0, device=device))

        self.total_replaced = 0

    def get_compression_params(self):
        """Return bit-depth parameters for optimizer."""
        return list(self.bit_depths.values())

    def compute_compression_loss(self):
        """Average bit-depth across all channels."""
        total, count = 0, 0
        for name, bits in self.bit_depths.items():
            total += bits.clamp(min=0).sum()
            count += bits.numel()
        return total / max(count, 1) if count > 0 else torch.tensor(0.0)

    def update_utility(self, features_dict):
        """Update utility estimates based on current activations."""
        with torch.no_grad():
            for name, module in self.conv_layers:
                self.ages[name] += 1

                if name not in features_dict:
                    continue

                feat = features_dict[name]  # (B, C, H, W)

                if self.utility_type == 'contribution':
                    # CBP's default: output_weight_mag * activation_mag
                    # Approximate with just activation magnitude (simpler, same idea)
                    act_mag = feat.abs().mean(dim=(0, 2, 3))  # per-channel
                    new_util = act_mag
                elif self.utility_type == 'compression':
                    # Use learned bit-depth as utility (higher = more important)
                    new_util = self.bit_depths[name].clamp(min=0).detach()
                elif self.utility_type == 'random':
                    # Random utility (control)
                    new_util = torch.rand(module.out_channels, device=self.device)
                else:
                    new_util = torch.ones(module.out_channels, device=self.device)

                self.utility[name] = self.decay_rate * self.utility[name] + (1 - self.decay_rate) * new_util

    def replace_units(self):
        """Replace lowest-utility units that are mature enough."""
        total_replaced = 0

        with torch.no_grad():
            for name, module in self.conv_layers:
                # Find eligible units (age > maturity_threshold)
                eligible = self.ages[name] > self.maturity_threshold
                eligible_indices = torch.where(eligible)[0]

                if eligible_indices.shape[0] == 0:
                    continue

                # How many to replace
                n_replace = self.replacement_rate * eligible_indices.shape[0]
                if n_replace < 1:
                    if torch.rand(1).item() < n_replace:
                        n_replace = 1
                    else:
                        continue
                n_replace = int(n_replace)

                # Find lowest utility among eligible
                eligible_utility = self.utility[name][eligible_indices]
                _, lowest_indices = torch.topk(-eligible_utility, min(n_replace, eligible_indices.shape[0]))
                to_replace = eligible_indices[lowest_indices]

                # Reinitialize: input weights random, output weights zero
                nn.init.kaiming_normal_(module.weight.data[to_replace], mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    module.bias.data[to_replace] = 0

                # Reset utility and age
                self.utility[name][to_replace] = 0
                self.ages[name][to_replace] = 0

                # Reset bit-depth for compression method
                if name in self.bit_depths:
                    self.bit_depths[name].data[to_replace] = 8.0

                total_replaced += n_replace

        self.total_replaced += total_replaced
        return total_replaced


def get_cifar100_data(data_root='/mnt/e/datasets/cifar100'):
    """Load CIFAR-100 with standard transforms."""
    train_transform = T.Compose([
        T.RandomCrop(32, padding=4),
        T.RandomHorizontalFlip(),
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])
    test_transform = T.Compose([
        T.ToTensor(),
        T.Normalize((0.5071, 0.4867, 0.4408), (0.2675, 0.2565, 0.2761)),
    ])

    train_data = torchvision.datasets.CIFAR100(root=data_root, train=True, download=False, transform=train_transform)
    test_data = torchvision.datasets.CIFAR100(root=data_root, train=False, download=False, transform=test_transform)
    return train_data, test_data


def get_class_subset(dataset, classes):
    """Get subset of dataset containing only specified classes."""
    targets = np.array(dataset.targets)
    indices = np.where(np.isin(targets, classes))[0]
    return Subset(dataset, indices)


def evaluate(model, test_loader, classes, device):
    """Evaluate on current set of classes."""
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for x, y in test_loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)[:, classes]
            # Remap labels to 0..len(classes)-1
            label_map = {c: i for i, c in enumerate(classes)}
            mapped_y = torch.tensor([label_map[yi.item()] for yi in y], device=device)
            correct += (logits.argmax(1) == mapped_y).sum().item()
            total += y.size(0)
    model.train()
    return correct / total if total > 0 else 0.0


def run_experiment(method, config, device='cuda'):
    """Run the incremental CIFAR-100 plasticity experiment."""
    torch.manual_seed(config['seed'])
    torch.cuda.manual_seed(config['seed'])
    np.random.seed(config['seed'])

    train_data, test_data = get_cifar100_data()
    model = SimpleResNet18(num_classes=100).to(device)

    # Random class order
    class_order = np.random.permutation(100).tolist()

    # Set up unit replacer
    replacer = None
    if method != 'standard':
        replacer = UnitReplacer(
            model, utility_type=method,
            replacement_rate=config.get('replacement_rate', 0.001),
            decay_rate=config.get('decay_rate', 0.99),
            maturity_threshold=config.get('maturity_threshold', 100),
            device=device,
        )

    # Optimizer
    params = list(model.parameters())
    if replacer and method == 'compression':
        comp_params = replacer.get_compression_params()
        optimizer = torch.optim.SGD([
            {'params': params, 'lr': config['lr'], 'momentum': 0.9, 'weight_decay': 5e-4},
        ])
        comp_optimizer = torch.optim.Adam(comp_params, lr=0.1)
    else:
        optimizer = torch.optim.SGD(params, lr=config['lr'], momentum=0.9, weight_decay=5e-4)
        comp_optimizer = None

    gamma = config.get('gamma', 0.01)
    loss_fn = nn.CrossEntropyLoss()

    # Training loop
    num_classes_now = 5
    class_increase_freq = config.get('class_increase_freq', 200)
    total_epochs = config.get('total_epochs', 4000)

    test_accs = []
    epoch_times = []

    for epoch in range(total_epochs):
        # Adjust LR per task
        task_epoch = epoch % class_increase_freq
        if task_epoch == 0:
            lr = config['lr']
        elif task_epoch == 60:
            lr = config['lr'] * 0.2
        elif task_epoch == 120:
            lr = config['lr'] * 0.04
        elif task_epoch == 160:
            lr = config['lr'] * 0.008
        else:
            lr = None
        if lr is not None:
            for g in optimizer.param_groups:
                g['lr'] = lr

        # Get current classes and data
        current_classes = class_order[:num_classes_now]
        train_subset = get_class_subset(train_data, current_classes)
        train_loader = DataLoader(train_subset, batch_size=128, shuffle=True, num_workers=4)

        # Train one epoch
        model.train()
        t0 = time.time()

        for x, y in train_loader:
            x, y = x.to(device), y.to(device)

            if replacer:
                out, features = model(x, return_features=True)
            else:
                out = model(x)

            logits = out[:, current_classes]
            label_map = {c: i for i, c in enumerate(current_classes)}
            mapped_y = torch.tensor([label_map[yi.item()] for yi in y], device=device)
            loss = loss_fn(logits, mapped_y)

            # Compression loss
            if method == 'compression' and comp_optimizer is not None:
                comp_loss = gamma * replacer.compute_compression_loss()
                loss = loss + comp_loss

            optimizer.zero_grad()
            if comp_optimizer:
                comp_optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            if comp_optimizer:
                comp_optimizer.step()

            # Update utility and replace units
            if replacer:
                # Build features dict
                layer_names = ['layer0', 'layer1', 'layer2', 'layer3', 'layer4']
                feat_dict = {}
                for name, module in replacer.conv_layers:
                    # Find which feature tensor corresponds to this layer
                    for i, ln in enumerate(layer_names):
                        if name.startswith(ln) or ln in name:
                            feat_dict[name] = features[i]
                            break
                replacer.update_utility(feat_dict)
                replacer.replace_units()

        epoch_time = time.time() - t0
        epoch_times.append(epoch_time)

        # Evaluate every 10 epochs
        if (epoch + 1) % 10 == 0:
            test_subset = get_class_subset(test_data, current_classes)
            test_loader = DataLoader(test_subset, batch_size=256, shuffle=False, num_workers=4)
            acc = evaluate(model, test_loader, current_classes, device)
            test_accs.append((epoch + 1, acc, num_classes_now))

            if (epoch + 1) % 50 == 0:
                replaced_str = f", replaced={replacer.total_replaced}" if replacer else ""
                print(f"  Epoch {epoch+1}/{total_epochs}: classes={num_classes_now}, "
                      f"acc={acc*100:.1f}%{replaced_str}")

        # Add new classes
        if (epoch + 1) % class_increase_freq == 0 and num_classes_now < 100:
            num_classes_now += 5
            print(f"  === Now {num_classes_now} classes ===")

    return test_accs


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='standard',
                        choices=['standard', 'contribution', 'compression', 'random'])
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--replacement_rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--total_epochs', type=int, default=4000)
    parser.add_argument('--class_increase_freq', type=int, default=200)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    device = 'cuda'
    config = vars(args)

    print(f"Plasticity experiment: method={args.method}, epochs={args.total_epochs}")
    accs = run_experiment(args.method, config, device)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'test_accs': accs, 'config': config},
               f'checkpoints/plasticity_{args.method}.pt')

    # Print final accuracy for each task
    print(f"\nFinal accuracies:")
    for epoch, acc, n_classes in accs[-20:]:
        print(f"  Epoch {epoch}: {n_classes} classes, acc={acc*100:.1f}%")


if __name__ == '__main__':
    main()
