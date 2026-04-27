"""ConvNeXt with differentiable quantisation for self-compression.

Block design follows Liu et al. 2022 ("A ConvNet for the 2020s") exactly:
  depthwise 7x7 conv -> LayerNorm -> Linear(4x expand) -> GELU -> Linear(project)
  -> LayerScale gamma -> (DropPath) -> residual add.

Stem is adapted for CIFAR (32x32) just like our ResNet variants (3x3 vs 7x7):
  - image_size <= 32: 2x2 stride-2 stem (32 -> 16), giving feature maps
    16 -> 8 -> 4 -> 2 across the four stages.
  - image_size > 32: original 4x4 stride-4 stem.

Quantisation placement (consistent with ResNet/ViT variants):
  - Stem 4x4 conv: NOT quantised (input layer).
  - Intermediate stage downsamplers: NOT quantised (analogous to ResNet 1x1
    shortcut downsample — shape-shifting, not feature processing).
  - Per-block depthwise 7x7 conv + both pointwise Linears: quantised.
  - Final LayerNorm + per-task heads: NOT quantised.

Multi-head classifier (one head per task) for class-incremental CL.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F

from .quantization import CompressionGranularity
from .resnet import QuantizedConv2d
from .mlp import QuantizedLinear


class _LayerNormCF(nn.Module):
    """LayerNorm over channel dim for (N, C, H, W) tensors. Mirrors the
    `data_format='channels_first'` branch of the official ConvNeXt LayerNorm."""

    def __init__(self, num_channels: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_channels))
        self.bias = nn.Parameter(torch.zeros(num_channels))
        self.eps = eps

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        return self.weight[:, None, None] * x + self.bias[:, None, None]


def _drop_path(x, drop_prob: float, training: bool):
    if drop_prob <= 0.0 or not training:
        return x
    keep = 1.0 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    mask = x.new_empty(shape).bernoulli_(keep) / keep
    return x * mask


class _Block(nn.Module):
    """ConvNeXt block. Implementation (2) from the official code: dwconv stays
    in (N, C, H, W); permute to (N, H, W, C) for LayerNorm + Linears, permute
    back. Channel-last is faster in PyTorch."""

    def __init__(self, dim: int, drop_path: float = 0.0,
                 layer_scale_init_value: float = 1e-6,
                 granularity=CompressionGranularity.CHANNEL,
                 group_size: int = 16, init_bit_depth: float = 8.0,
                 quantize: bool = True):
        super().__init__()
        self.dwconv = QuantizedConv2d(dim, dim, kernel_size=7, padding=3,
                                      groups=dim, bias=True,
                                      granularity=granularity,
                                      group_size=group_size,
                                      init_bit_depth=init_bit_depth,
                                      quantize=quantize)

        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = QuantizedLinear(dim, 4 * dim, granularity=granularity,
                                       group_size=group_size,
                                       init_bit_depth=init_bit_depth, quantize=quantize)
        self.act = nn.GELU()
        self.pwconv2 = QuantizedLinear(4 * dim, dim, granularity=granularity,
                                       group_size=group_size,
                                       init_bit_depth=init_bit_depth, quantize=quantize)
        if layer_scale_init_value > 0:
            self.gamma = nn.Parameter(layer_scale_init_value * torch.ones(dim))
        else:
            self.gamma = None
        self.drop_path = drop_path

    def forward(self, x):
        inp = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        return inp + _drop_path(x, self.drop_path, self.training)


class QuantizedConvNeXt(nn.Module):
    """ConvNeXt with quantised dw + pw layers and per-task heads."""

    def __init__(self, num_classes_per_task: int = 10, num_tasks: int = 10,
                 depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                 drop_path_rate: float = 0.0, layer_scale_init_value: float = 1e-6,
                 granularity=CompressionGranularity.CHANNEL, group_size: int = 16,
                 init_bit_depth: float = 8.0, single_head: bool = False,
                 image_size: int = 32, quantize: bool = True):
        super().__init__()
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.single_head = single_head
        self.image_size = image_size

        # Stem: CIFAR adapts 4x4-s4 to 2x2-s2 to keep enough spatial resolution.
        if image_size <= 32:
            stem_conv = nn.Conv2d(3, dims[0], kernel_size=2, stride=2)
        else:
            stem_conv = nn.Conv2d(3, dims[0], kernel_size=4, stride=4)
        self.downsample_layers = nn.ModuleList([
            nn.Sequential(stem_conv, _LayerNormCF(dims[0], eps=1e-6))
        ])
        for i in range(3):
            self.downsample_layers.append(nn.Sequential(
                _LayerNormCF(dims[i], eps=1e-6),
                nn.Conv2d(dims[i], dims[i + 1], kernel_size=2, stride=2),
            ))

        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        self.stages = nn.ModuleList()
        cur = 0
        for i in range(4):
            stage = nn.Sequential(*[
                _Block(dim=dims[i], drop_path=dp_rates[cur + j],
                       layer_scale_init_value=layer_scale_init_value,
                       granularity=granularity, group_size=group_size,
                       init_bit_depth=init_bit_depth, quantize=quantize)
                for j in range(depths[i])
            ])
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)
        feat_dim = dims[-1]
        if single_head:
            self.classifier = nn.Linear(feat_dim, num_classes_per_task * num_tasks)
        else:
            self.heads = nn.ModuleList([
                nn.Linear(feat_dim, num_classes_per_task) for _ in range(num_tasks)
            ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def _features(self, x):
        for ds, st in zip(self.downsample_layers, self.stages):
            x = st(ds(x))
        return self.norm(x.mean([-2, -1]))

    def forward(self, x, task_id=None):
        h = self._features(x)
        if self.single_head:
            return self.classifier(h)
        if task_id is not None:
            return self.heads[task_id](h)
        return torch.cat([head(h) for head in self.heads], dim=1)


def convnext_tiny(num_classes_per_task=10, num_tasks=10, quantize=True, **kw):
    return QuantizedConvNeXt(num_classes_per_task=num_classes_per_task,
                             num_tasks=num_tasks,
                             depths=(3, 3, 9, 3), dims=(96, 192, 384, 768),
                             quantize=quantize, **kw)
