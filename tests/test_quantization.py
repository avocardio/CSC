"""Tests for the differentiable quantization module."""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
from models.quantization import (
    quantize, ste_round, DifferentiableQuantizer, CompressionGranularity,
    compute_average_bit_depth, get_compression_stats,
)
from models.resnet import QuantizedResNet18, QuantizedConv2d


def test_ste_round():
    x = torch.tensor([1.3, 2.7, -0.5, 0.0], requires_grad=True)
    y = ste_round(x)
    assert torch.equal(y, torch.tensor([1.0, 3.0, 0.0, 0.0]))
    # STE: gradients pass through
    y.sum().backward()
    assert torch.equal(x.grad, torch.ones(4))
    print("  ste_round: OK")


def test_quantize_basic():
    x = torch.randn(4, 3, 3, 3)
    b = torch.tensor([8.0])  # 8-bit
    e = torch.tensor([-4.0])
    q = quantize(x, b, e)
    # With 8 bits and exponent -4, quantized values should be close to original
    assert q.shape == x.shape
    assert (q - x).abs().max() < 0.1  # rough check
    print("  quantize basic: OK")


def test_quantize_zero_bits():
    x = torch.randn(4, 3, 3, 3)
    b = torch.tensor([0.0])
    e = torch.tensor([-4.0])
    q = quantize(x, b, e)
    # b=0 should zero everything: range is [-2^(-1), 2^(-1)-1] = [-0.5, -0.5]
    # After clamping to -0.5 and rounding -> -1 or 0
    # Actually when b=0: half_range = 2^(-1) = 0.5, lower = -0.5, upper = -0.5
    # Everything clamps to -0.5, rounds to 0
    # Then scaled by 2^e
    # Let's just verify output is very small
    assert q.abs().max() < 0.1, f"Expected near-zero output with b=0, got max={q.abs().max()}"
    print("  quantize zero bits: OK")


def test_quantize_gradient():
    x = torch.randn(4, 3, 3, 3, requires_grad=True)
    b = torch.tensor([8.0], requires_grad=True)
    e = torch.tensor([-4.0], requires_grad=True)
    q = quantize(x, b, e)
    loss = q.sum()
    loss.backward()
    assert x.grad is not None
    assert b.grad is not None
    assert e.grad is not None
    print("  quantize gradient: OK")


def test_differentiable_quantizer_channel():
    shape = (16, 8, 3, 3)
    dq = DifferentiableQuantizer(shape, CompressionGranularity.CHANNEL)
    w = torch.randn(shape)
    q = dq(w)
    assert q.shape == shape
    bits = dq.compute_layer_bits()
    assert bits > 0
    print(f"  DifferentiableQuantizer channel: OK (bits={bits.item():.0f})")


def test_differentiable_quantizer_group():
    shape = (16, 8, 3, 3)
    dq = DifferentiableQuantizer(shape, CompressionGranularity.GROUP, group_size=16)
    w = torch.randn(shape)
    q = dq(w)
    assert q.shape == shape
    print("  DifferentiableQuantizer group: OK")


def test_differentiable_quantizer_weight():
    shape = (16, 8, 3, 3)
    dq = DifferentiableQuantizer(shape, CompressionGranularity.WEIGHT)
    w = torch.randn(shape)
    q = dq(w)
    assert q.shape == shape
    print("  DifferentiableQuantizer weight: OK")


def test_quantized_resnet18():
    model = QuantizedResNet18(num_classes_per_task=10, num_tasks=2,
                              granularity=CompressionGranularity.CHANNEL)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, task_id=0)
    assert out.shape == (2, 10)
    out2 = model(x, task_id=1)
    assert out2.shape == (2, 10)
    print("  QuantizedResNet18 forward: OK")


def test_resnet18_compression_stats():
    model = QuantizedResNet18(num_classes_per_task=10, num_tasks=1,
                              granularity=CompressionGranularity.CHANNEL)
    stats = get_compression_stats(model)
    print(f"  ResNet18 stats: {stats['total_weights']:,} weights, "
          f"{stats['total_channels']} channels, "
          f"avg bit depth={stats['avg_bit_depth']:.1f}")
    assert stats['avg_bit_depth'] > 0


def test_resnet18_backward():
    model = QuantizedResNet18(num_classes_per_task=10, num_tasks=1,
                              granularity=CompressionGranularity.CHANNEL)
    x = torch.randn(2, 3, 32, 32)
    out = model(x, task_id=0)
    loss = out.sum()
    Q = compute_average_bit_depth(model)
    total = loss + 0.01 * Q
    total.backward()
    # Check gradients exist on quantization params
    from models.quantization import get_quantizers
    for q in get_quantizers(model):
        assert q.bit_depth.grad is not None, "No gradient on bit_depth"
        assert q.exponent.grad is not None, "No gradient on exponent"
    print("  ResNet18 backward with compression loss: OK")


def test_resnet18_cuda():
    if not torch.cuda.is_available():
        print("  CUDA not available, skipping")
        return
    model = QuantizedResNet18(num_classes_per_task=10, num_tasks=1).cuda()
    x = torch.randn(4, 3, 32, 32).cuda()
    with torch.amp.autocast('cuda'):
        out = model(x, task_id=0)
        Q = compute_average_bit_depth(model)
        loss = out.sum() + 0.01 * Q
    loss.backward()
    print(f"  ResNet18 CUDA + AMP: OK (output shape={out.shape})")


if __name__ == '__main__':
    print("Testing quantization module...")
    test_ste_round()
    test_quantize_basic()
    test_quantize_zero_bits()
    test_quantize_gradient()
    test_differentiable_quantizer_channel()
    test_differentiable_quantizer_group()
    test_differentiable_quantizer_weight()
    test_quantized_resnet18()
    test_resnet18_compression_stats()
    test_resnet18_backward()
    test_resnet18_cuda()
    print("\nAll tests passed!")
