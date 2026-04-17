"""Pretrained ResNet-18 with optional differentiable quantization for CSC.

Loads ImageNet-pretrained weights, adds multi-head classifier and
optional quantization wrappers on conv layers.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from .quantization import DifferentiableQuantizer, CompressionGranularity


class QuantizedConvWrapper(nn.Module):
    """Wraps an existing Conv2d with differentiable quantization."""

    def __init__(self, conv, granularity=CompressionGranularity.CHANNEL,
                 group_size=16, init_bit_depth=8.0):
        super().__init__()
        self.conv = conv
        self.quantizer = DifferentiableQuantizer(
            conv.weight.shape,
            granularity=granularity,
            group_size=group_size,
            init_bit_depth=init_bit_depth,
        )

    def forward(self, x):
        q_weight = self.quantizer(self.conv.weight)
        return F.conv2d(x, q_weight, self.conv.bias,
                        self.conv.stride, self.conv.padding,
                        self.conv.dilation, self.conv.groups)


def wrap_conv_layers(model, granularity=CompressionGranularity.CHANNEL,
                     group_size=16, init_bit_depth=8.0, skip_first=True):
    """Recursively wrap all Conv2d layers with quantization.

    Args:
        model: nn.Module to wrap
        skip_first: if True, skip the very first conv layer (keep full precision)
    """
    first_skipped = [not skip_first]  # mutable to track across recursion

    def _wrap(module, name=''):
        for child_name, child in module.named_children():
            full_name = f"{name}.{child_name}" if name else child_name
            if isinstance(child, nn.Conv2d):
                if not first_skipped[0]:
                    first_skipped[0] = True
                    continue  # skip first conv
                wrapped = QuantizedConvWrapper(
                    child, granularity=granularity,
                    group_size=group_size, init_bit_depth=init_bit_depth)
                setattr(module, child_name, wrapped)
            else:
                _wrap(child, full_name)

    _wrap(model)


class PretrainedResNet18CL(nn.Module):
    """Pretrained ResNet-18 for continual learning with optional quantization."""

    def __init__(self, num_classes_per_task=10, num_tasks=10,
                 granularity=CompressionGranularity.CHANNEL,
                 init_bit_depth=8.0, quantize=True, freeze_backbone=False):
        super().__init__()
        # Load pretrained backbone
        backbone = models.resnet18(weights='IMAGENET1K_V1')

        # Remove original FC layer
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool

        # Add quantization wrappers
        if quantize:
            wrap_conv_layers(self.features, granularity=granularity,
                             init_bit_depth=init_bit_depth, skip_first=True)

        # Optionally freeze backbone (only train heads + quantization params)
        if freeze_backbone:
            for name, param in self.features.named_parameters():
                if 'quantizer' not in name:
                    param.requires_grad = False

        # Multi-head classifier
        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        # Initialize heads
        for head in self.heads:
            nn.init.normal_(head.weight, 0, 0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, x, task_id=None):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)

    def get_features(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class PretrainedResNet18Plain(nn.Module):
    """Pretrained ResNet-18 without quantization (for baselines)."""

    def __init__(self, num_classes_per_task=10, num_tasks=10, freeze_backbone=False):
        super().__init__()
        backbone = models.resnet18(weights='IMAGENET1K_V1')
        self.features = nn.Sequential(
            backbone.conv1, backbone.bn1, backbone.relu, backbone.maxpool,
            backbone.layer1, backbone.layer2, backbone.layer3, backbone.layer4,
        )
        self.avgpool = backbone.avgpool

        if freeze_backbone:
            for param in self.features.parameters():
                param.requires_grad = False

        self.heads = nn.ModuleList([
            nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
        ])
        self.num_tasks = num_tasks

        for head in self.heads:
            nn.init.normal_(head.weight, 0, 0.01)
            nn.init.constant_(head.bias, 0)

    def forward(self, x, task_id=None):
        x = self.features(x)
        x = self.avgpool(x)
        features = torch.flatten(x, 1)
        if task_id is not None:
            return self.heads[task_id](features)
        return torch.cat([h(features) for h in self.heads], dim=1)
