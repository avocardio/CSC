"""Simple MLP for Permuted MNIST experiments.

Standard architecture for CL benchmarks: 784 -> 256 -> 256 -> 10.
Supports quantized (CSC) and plain (baseline) variants.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from .quantization import DifferentiableQuantizer, CompressionGranularity


class QuantizedLinear(nn.Module):
    """Linear layer with differentiable quantization."""

    def __init__(self, in_features, out_features, granularity=CompressionGranularity.CHANNEL,
                 group_size=16, init_bit_depth=8.0, quantize=True):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)
        self.do_quantize = quantize
        if quantize:
            self.quantizer = DifferentiableQuantizer(
                self.linear.weight.shape,
                granularity=granularity,
                group_size=group_size,
                init_bit_depth=init_bit_depth,
            )

    def forward(self, x):
        if self.do_quantize:
            q_weight = self.quantizer(self.linear.weight)
            return F.linear(x, q_weight, self.linear.bias)
        return self.linear(x)

    @property
    def weight(self):
        return self.linear.weight

    @property
    def bias(self):
        return self.linear.bias

    @property
    def out_features(self):
        return self.linear.out_features


class QuantizedMLP(nn.Module):
    """MLP with differentiable quantization and multi-head output."""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10,
                 num_tasks=10, granularity=CompressionGranularity.CHANNEL,
                 init_bit_depth=8.0):
        super().__init__()
        self.fc1 = QuantizedLinear(input_size, hidden_size, granularity=granularity,
                                   init_bit_depth=init_bit_depth)
        self.fc2 = QuantizedLinear(hidden_size, hidden_size, granularity=granularity,
                                   init_bit_depth=init_bit_depth)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_tasks)
        ])
        self.num_tasks = num_tasks

    def forward(self, x, task_id=None):
        x = x.view(x.size(0), -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if task_id is not None:
            return self.heads[task_id](x)
        return torch.cat([h(x) for h in self.heads], dim=1)


class SimpleMLP(nn.Module):
    """Plain MLP for baselines."""

    def __init__(self, input_size=784, hidden_size=256, num_classes=10, num_tasks=10):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.heads = nn.ModuleList([
            nn.Linear(hidden_size, num_classes) for _ in range(num_tasks)
        ])
        self.num_tasks = num_tasks

    def forward(self, x, task_id=None):
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        if task_id is not None:
            return self.heads[task_id](x)
        return torch.cat([h(x) for h in self.heads], dim=1)
