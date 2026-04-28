"""Post-hoc uniform quantization of a saved model state, then re-eval per-task.

Tests whether naive INT-N PTQ of a DER++/EWC-trained model already matches the
CSC headline. If yes, CSC's "fewer bits at same accuracy" claim collapses.

Per-channel symmetric uniform quantization of conv/linear weights:
    scale_c = max_abs(w_c) / (2^(b-1) - 1)
    w_q     = scale_c * round(w_c / scale_c).clamp(-(2^(b-1)-1), 2^(b-1)-1)

Skips: BatchNorm, downsample 1x1 convs (matching what CSC also skips), heads.

Usage: python analysis/ptq_baseline.py --ckpt path.pt --bits 8 --dataset cifar100
"""
from __future__ import annotations
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import numpy as np
from copy import deepcopy

from models.resnet import (QuantizedResNet18, QuantizedResNet50,
                           QuantizedResNet101, QuantizedConv2d)
from models.convnext import convnext_tiny
from models.mlp import QuantizedMLP, QuantizedLinear as QuantizedLinearMLP
from data.split_cifar100 import SplitCIFAR100
from data.permuted_mnist import PermutedMNIST
from training.metrics import evaluate_all_tasks


def build_model(arch: str, num_tasks: int, classes_per_task: int, quantize=False):
    if arch == 'resnet18':
        return QuantizedResNet18(num_classes_per_task=classes_per_task,
                                 num_tasks=num_tasks, quantize=quantize)
    if arch == 'resnet50':
        return QuantizedResNet50(num_classes_per_task=classes_per_task,
                                 num_tasks=num_tasks, quantize=quantize)
    if arch == 'resnet101':
        return QuantizedResNet101(num_classes_per_task=classes_per_task,
                                  num_tasks=num_tasks, quantize=quantize)
    if arch == 'convnext_tiny':
        return convnext_tiny(num_classes_per_task=classes_per_task,
                             num_tasks=num_tasks, quantize=quantize)
    if arch == 'mlp':
        return QuantizedMLP(num_tasks=1, quantize=quantize)
    raise ValueError(arch)


@torch.no_grad()
def quantize_per_channel_symmetric(W: torch.Tensor, bits: int):
    """Per-output-channel symmetric uniform quantization.
    W: (O, ...) tensor; quantizes along axis-0."""
    if bits >= 32:
        return W
    qmax = 2 ** (bits - 1) - 1
    if qmax < 1:
        # Special case b=1: ternary {-s, 0, +s}.
        qmax = 1
    O = W.shape[0]
    flat = W.reshape(O, -1)
    max_abs = flat.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
    scale = max_abs / qmax
    q = torch.round(flat / scale).clamp(-qmax, qmax)
    out = (q * scale).reshape(W.shape)
    return out


@torch.no_grad()
def apply_ptq(model: nn.Module, bits: int):
    """Quantize the *same* layers CSC would (conv weights inside residual blocks
    + linear weights inside ConvNeXt blocks / MLPs), at b bits per channel.
    Skips stems, downsample 1x1 convs, classifier heads."""
    n_quantized = 0
    for name, m in model.named_modules():
        # Skip heads / classifier modules
        if 'heads' in name or 'classifier' in name or 'fc' in name and name.endswith('fc'):
            continue
        # Conv inside QuantizedConv2d wrappers (both bottleneck and basic blocks).
        if isinstance(m, QuantizedConv2d):
            m.conv.weight.data = quantize_per_channel_symmetric(m.conv.weight.data, bits)
            n_quantized += 1
            continue
        # QuantizedLinear (MLP / ConvNeXt pwconvs).
        if isinstance(m, QuantizedLinearMLP):
            m.linear.weight.data = quantize_per_channel_symmetric(m.linear.weight.data, bits)
            n_quantized += 1
            continue
    return n_quantized


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='_final.pt saved by run_supervised.py')
    p.add_argument('--bits', type=int, default=8, help='target bit-depth for PTQ')
    p.add_argument('--data_root', default='/mnt/e/datasets/cifar100')
    p.add_argument('--batch_size', type=int, default=128)
    args = p.parse_args()

    s = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    arch = s['model_arch']
    dataset = s['dataset']
    num_tasks = s['num_tasks']
    classes_per_task = s['classes_per_task']
    seed = s['seed']
    method = s['method']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    # Benchmark for evaluation only (test loaders).
    if dataset == 'cifar100':
        bench = SplitCIFAR100(data_root=args.data_root, num_tasks=num_tasks,
                              batch_size=args.batch_size, seed=seed)
    else:
        bench = PermutedMNIST(data_root=args.data_root, num_tasks=num_tasks,
                              batch_size=args.batch_size, seed=seed)

    model = build_model(arch, num_tasks, classes_per_task, quantize=False).to(device)
    model.load_state_dict(s['model'], strict=False)
    model.eval()

    # FP32 baseline accuracy
    fp32_accs = evaluate_all_tasks(model, bench, num_tasks, device)
    fp32_avg = float(np.mean(fp32_accs))
    print(f'FP32 baseline: avg = {fp32_avg*100:.2f}%')
    for j, a in enumerate(fp32_accs):
        print(f'  Task {j}: {a*100:.2f}%')

    # PTQ at requested bits
    print(f'\n--- Applying PTQ at b = {args.bits} bits ---')
    nq = apply_ptq(model, args.bits)
    print(f'  Quantized {nq} layers')

    ptq_accs = evaluate_all_tasks(model, bench, num_tasks, device)
    ptq_avg = float(np.mean(ptq_accs))
    print(f'PTQ b={args.bits}: avg = {ptq_avg*100:.2f}%')
    for j, a in enumerate(ptq_accs):
        print(f'  Task {j}: {a*100:.2f}%')

    out = args.ckpt.replace('.pt', f'_ptq{args.bits}.json')
    with open(out, 'w') as f:
        json.dump({
            'method': method,
            'arch': arch,
            'seed': seed,
            'num_tasks': num_tasks,
            'fp32_avg': fp32_avg,
            'fp32_per_task': [float(a) for a in fp32_accs],
            'ptq_bits': args.bits,
            'ptq_avg': ptq_avg,
            'ptq_per_task': [float(a) for a in ptq_accs],
        }, f, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
