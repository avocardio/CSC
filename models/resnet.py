"""ResNet-18 with differentiable quantization for self-compression.

Key design decisions:
- Quantization applied to all conv layers
- Residual connections: we DON'T compress skip connection channels to avoid
  dimension mismatches. Only compress within blocks.
- For the shortcut 1x1 convs (in blocks where dimensions change), we also
  don't compress them to maintain skip connection integrity.
- Multi-head classifier for continual learning (one head per task).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import DifferentiableQuantizer, CompressionGranularity


class QuantizedConv2d(nn.Module):
    """Conv2d with differentiable quantization on weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, bias=False, granularity=CompressionGranularity.CHANNEL,
                 group_size=16, init_bit_depth=8.0, quantize=True):
        super().__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size,
                              stride=stride, padding=padding, bias=bias)
        self.do_quantize = quantize
        if quantize:
            self.quantizer = DifferentiableQuantizer(
                self.conv.weight.shape,
                granularity=granularity,
                group_size=group_size,
                init_bit_depth=init_bit_depth,
            )

    def forward(self, x):
        if self.do_quantize:
            q_weight = self.quantizer(self.conv.weight)
            return F.conv2d(x, q_weight, self.conv.bias,
                           self.conv.stride, self.conv.padding)
        return self.conv(x)

    @property
    def weight(self):
        return self.conv.weight

    @weight.setter
    def weight(self, val):
        self.conv.weight = val

    @property
    def bias(self):
        return self.conv.bias

    @property
    def out_channels(self):
        return self.conv.out_channels

    @property
    def in_channels(self):
        return self.conv.in_channels


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1, downsample=None,
                 granularity=CompressionGranularity.CHANNEL, group_size=16,
                 init_bit_depth=8.0, quantize=True):
        super().__init__()
        self.conv1 = QuantizedConv2d(in_channels, out_channels, 3,
                                     stride=stride, padding=1, bias=False,
                                     granularity=granularity, group_size=group_size,
                                     init_bit_depth=init_bit_depth, quantize=quantize)
        self.bn1 = nn.BatchNorm2d(out_channels)

        self.conv2 = QuantizedConv2d(out_channels, out_channels, 3,
                                     stride=1, padding=1, bias=False,
                                     granularity=granularity, group_size=group_size,
                                     init_bit_depth=init_bit_depth, quantize=quantize)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = F.relu(out)
        return out


class Bottleneck(nn.Module):
    """ResNet-50 bottleneck block (1x1 -> 3x3 -> 1x1, 4x expansion).
    All three convs are wrapped with QuantizedConv2d when quantize=True."""
    expansion = 4

    def __init__(self, in_channels, planes, stride=1, downsample=None,
                 granularity=CompressionGranularity.CHANNEL, group_size=16,
                 init_bit_depth=8.0, quantize=True):
        super().__init__()
        self.conv1 = QuantizedConv2d(in_channels, planes, 1, bias=False,
                                     granularity=granularity, group_size=group_size,
                                     init_bit_depth=init_bit_depth, quantize=quantize)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = QuantizedConv2d(planes, planes, 3, stride=stride, padding=1,
                                     bias=False, granularity=granularity,
                                     group_size=group_size, init_bit_depth=init_bit_depth,
                                     quantize=quantize)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = QuantizedConv2d(planes, planes * self.expansion, 1, bias=False,
                                     granularity=granularity, group_size=group_size,
                                     init_bit_depth=init_bit_depth, quantize=quantize)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return F.relu(out)


class QuantizedResNet18(nn.Module):

    """ResNet-18 with differentiable quantization and multi-head classifier."""

    def __init__(self, num_classes_per_task=10, num_tasks=10,
                 granularity=CompressionGranularity.CHANNEL, group_size=16,
                 init_bit_depth=8.0, single_head=False, image_size=32,
                 quantize=True):
        super().__init__()
        self.granularity = granularity
        self.group_size = group_size
        self.init_bit_depth = init_bit_depth
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.single_head = single_head
        self.in_channels = 64
        self.image_size = image_size
        self.quantize_enabled = quantize

        # Initial conv (not compressed — first layer, keep full precision)
        if image_size <= 32:
            # CIFAR: 3x3 conv, no maxpool
            self.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
            self.maxpool = nn.Identity()
        else:
            # TinyImageNet/ImageNet: 7x7 conv stride 2 + maxpool
            self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
            self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(64)

        # Residual layers
        self.layer1 = self._make_layer(64, 2, stride=1)
        self.layer2 = self._make_layer(128, 2, stride=2)
        self.layer3 = self._make_layer(256, 2, stride=2)
        self.layer4 = self._make_layer(512, 2, stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        # Multi-head or single-head classifier
        if single_head:
            self.classifier = nn.Linear(512, num_classes_per_task * num_tasks)
        else:
            self.heads = nn.ModuleList([
                nn.Linear(512, num_classes_per_task) for _ in range(num_tasks)
            ])

        self._initialize_weights()

    def _make_layer(self, out_channels, num_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            # Downsample with uncompressed 1x1 conv to maintain skip connection dims
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )

        layers = []
        layers.append(BasicBlock(self.in_channels, out_channels, stride, downsample,
                                 self.granularity, self.group_size, self.init_bit_depth,
                                 quantize=self.quantize_enabled))
        self.in_channels = out_channels
        for _ in range(1, num_blocks):
            layers.append(BasicBlock(out_channels, out_channels, 1, None,
                                     self.granularity, self.group_size, self.init_bit_depth,
                                     quantize=self.quantize_enabled))

        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, task_id=None):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        features = torch.flatten(x, 1)

        if self.single_head:
            return self.classifier(features)
        else:
            if task_id is not None:
                return self.heads[task_id](features)
            else:
                # Return all heads' outputs concatenated
                outputs = [head(features) for head in self.heads]
                return torch.cat(outputs, dim=1)

    def get_features(self, x):
        """Extract features before classification head."""
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        return torch.flatten(x, 1)

    def add_task_head(self, num_classes=None):
        """Add a new classification head for a new task."""
        if self.single_head:
            return
        if num_classes is None:
            num_classes = self.num_classes_per_task
        device = next(self.parameters()).device
        new_head = nn.Linear(512, num_classes).to(device)
        nn.init.normal_(new_head.weight, 0, 0.01)
        nn.init.constant_(new_head.bias, 0)
        self.heads.append(new_head)
        self.num_tasks += 1


# ============================================================
# ResNet-50 (Bottleneck) — for the model-size scaling experiment
# ============================================================
class QuantizedResNet50(nn.Module):
    """ResNet-50 with channel-level differentiable quantisation, multi-head."""

    def __init__(self, num_classes_per_task=10, num_tasks=10,
                 granularity=CompressionGranularity.CHANNEL, group_size=16,
                 init_bit_depth=8.0, single_head=False, image_size=32,
                 quantize=True):
        super().__init__()
        self.granularity = granularity
        self.group_size = group_size
        self.init_bit_depth = init_bit_depth
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task
        self.single_head = single_head
        self.in_channels = 64
        self.image_size = image_size
        self.quantize_enabled = quantize

        if image_size <= 32:
            self.conv1 = nn.Conv2d(3, 64, 3, 1, 1, bias=False)
            self.maxpool = nn.Identity()
        else:
            self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
            self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.bn1 = nn.BatchNorm2d(64)

        self.layer1 = self._make_layer(64, 3, stride=1)
        self.layer2 = self._make_layer(128, 4, stride=2)
        self.layer3 = self._make_layer(256, 6, stride=2)
        self.layer4 = self._make_layer(512, 3, stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        feat_dim = 512 * Bottleneck.expansion                # 2048
        if single_head:
            self.classifier = nn.Linear(feat_dim, num_classes_per_task * num_tasks)
        else:
            self.heads = nn.ModuleList(
                [nn.Linear(feat_dim, num_classes_per_task) for _ in range(num_tasks)])
        self._init_weights()

    def _make_layer(self, planes, n_blocks, stride):
        downsample = None
        if stride != 1 or self.in_channels != planes * Bottleneck.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, planes * Bottleneck.expansion, 1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(planes * Bottleneck.expansion),
            )
        layers = [Bottleneck(self.in_channels, planes, stride, downsample,
                             self.granularity, self.group_size, self.init_bit_depth,
                             quantize=self.quantize_enabled)]
        self.in_channels = planes * Bottleneck.expansion
        for _ in range(1, n_blocks):
            layers.append(Bottleneck(self.in_channels, planes, 1, None,
                                     self.granularity, self.group_size,
                                     self.init_bit_depth,
                                     quantize=self.quantize_enabled))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1); nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01); nn.init.constant_(m.bias, 0)

    def forward(self, x, task_id=None):
        x = self.maxpool(F.relu(self.bn1(self.conv1(x))))
        x = self.layer1(x); x = self.layer2(x)
        x = self.layer3(x); x = self.layer4(x)
        x = self.avgpool(x)
        h = torch.flatten(x, 1)
        if self.single_head:
            return self.classifier(h)
        if task_id is not None:
            return self.heads[task_id](h)
        return torch.cat([head(h) for head in self.heads], dim=1)
