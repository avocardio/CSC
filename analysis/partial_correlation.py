"""Partial Spearman correlation rho(bit-depth, Fisher | |w|) for the CSC paper.

The Fisher correlation result rho(b, F) ≈ 0.71 may be a magnitude confound:
both bit-depth and empirical Fisher correlate with weight magnitude.

For each saved CSC model, this script:
  1. Reads per-channel bit-depth from the model's quantizer state.
  2. Computes per-channel weight magnitude (max |w| per output channel).
  3. Computes per-channel empirical Fisher on a small calibration subset.
  4. Reports rho(b, F), rho(b, |w|), rho(F, |w|), and partial rho(b, F | |w|).

Usage: python analysis/partial_correlation.py --ckpt path_to_csc_final.pt
"""
from __future__ import annotations
import os, sys, json, argparse, glob
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from scipy.stats import spearmanr, rankdata

from models.resnet import (QuantizedResNet18, QuantizedResNet50,
                           QuantizedResNet101, QuantizedConv2d)
from models.convnext import convnext_tiny
from models.mlp import QuantizedMLP, QuantizedLinear as QuantizedLinearMLP
from data.split_cifar100 import SplitCIFAR100
from data.permuted_mnist import PermutedMNIST


def build_model(arch: str, num_tasks: int, classes_per_task: int, quantize=True):
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


def per_channel_bit_depths(model: nn.Module) -> dict[str, np.ndarray]:
    """Return dict layer_name -> (out_channels,) array of per-channel bit-depths.
    Only quantized modules with do_quantize=True are returned."""
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, (QuantizedConv2d, QuantizedLinearMLP)):
            continue
        if not getattr(m, 'do_quantize', False):
            continue
        b = m.quantizer.get_channel_bit_depths().detach().clamp(min=0).cpu().numpy()
        out[name] = b
    return out


def per_channel_weight_magnitude(model: nn.Module) -> dict[str, np.ndarray]:
    """Per-channel max-abs weight magnitude (proxy for activation/feature scale)."""
    out = {}
    for name, m in model.named_modules():
        if not isinstance(m, (QuantizedConv2d, QuantizedLinearMLP)):
            continue
        if not getattr(m, 'do_quantize', False):
            continue
        if isinstance(m, QuantizedConv2d):
            W = m.conv.weight.detach()
        else:
            W = m.linear.weight.detach()
        flat = W.reshape(W.shape[0], -1).abs()
        out[name] = flat.amax(dim=1).cpu().numpy()
    return out


def per_channel_empirical_fisher(model: nn.Module, loader, task_id: int,
                                 device: str, n_batches: int = 8) -> dict[str, np.ndarray]:
    """Empirical diag-Fisher per output channel.

    F_c = E_x [ sum_{params in channel c} (d log p / d theta)^2 ]
    averaged over examples in the calibration set."""
    model = model.to(device)
    model.eval()

    # Collect parameters per quantized module (track by qualified name).
    chan_modules = {}
    for name, m in model.named_modules():
        if not isinstance(m, (QuantizedConv2d, QuantizedLinearMLP)):
            continue
        if not getattr(m, 'do_quantize', False):
            continue
        if isinstance(m, QuantizedConv2d):
            chan_modules[name] = m.conv.weight
        else:
            chan_modules[name] = m.linear.weight

    F_acc = {n: torch.zeros_like(p[:, ...].sum(dim=tuple(range(1, p.dim()))))
             for n, p in chan_modules.items()}
    n_done = 0
    for k, batch in enumerate(loader):
        if k >= n_batches:
            break
        x = batch[0].to(device); y = batch[1].to(device)
        for i in range(x.shape[0]):
            model.zero_grad(set_to_none=True)
            xi = x[i:i+1]; yi = y[i:i+1]
            logits = model(xi, task_id=task_id)
            logp = -F.cross_entropy(logits, yi)
            logp.backward()
            for n, p in chan_modules.items():
                if p.grad is None:
                    continue
                g = p.grad.detach()
                # Sum squared grads within each channel -> (out_channels,)
                g_sq = g.pow(2).reshape(g.shape[0], -1).sum(dim=1)
                F_acc[n] = F_acc[n] + g_sq
            n_done += 1
    # Average
    return {n: (v / max(n_done, 1)).cpu().numpy() for n, v in F_acc.items()}


def partial_spearman(x, y, z):
    """Partial Spearman: ρ(x, y | z). Compute by linearly regressing out
    rank(z) from rank(x) and rank(y), then correlating residuals."""
    rx = rankdata(x)
    ry = rankdata(y)
    rz = rankdata(z)
    rz_centered = rz - rz.mean()
    # Project out rz from rx and ry
    bx = (rx * rz_centered).sum() / (rz_centered ** 2).sum()
    by = (ry * rz_centered).sum() / (rz_centered ** 2).sum()
    rx_resid = rx - bx * rz_centered
    ry_resid = ry - by * rz_centered
    return float(np.corrcoef(rx_resid, ry_resid)[0, 1])


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='Saved CSC final.pt')
    p.add_argument('--data_root', default='/mnt/e/datasets/cifar100')
    p.add_argument('--task_id', type=int, default=0)
    p.add_argument('--n_batches', type=int, default=8)
    args = p.parse_args()

    s = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    arch = s['model_arch']; dataset = s['dataset']
    num_tasks = s['num_tasks']; classes_per_task = s['classes_per_task']
    seed = s['seed']

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = build_model(arch, num_tasks, classes_per_task, quantize=True)
    model.load_state_dict(s['model'])
    model.to(device)

    if dataset == 'cifar100':
        bench = SplitCIFAR100(data_root=args.data_root, num_tasks=num_tasks,
                              batch_size=64, seed=seed)
    else:
        bench = PermutedMNIST(data_root=args.data_root, num_tasks=num_tasks,
                              batch_size=64, seed=seed)
    train_loader, _ = bench.get_task_dataloaders(args.task_id)

    bd = per_channel_bit_depths(model)
    wm = per_channel_weight_magnitude(model)
    fi = per_channel_empirical_fisher(model, train_loader, args.task_id,
                                      device, n_batches=args.n_batches)

    # Concatenate across all quantized layers
    keys = sorted(bd.keys())
    b = np.concatenate([bd[k] for k in keys])
    w = np.concatenate([wm[k] for k in keys])
    f = np.concatenate([fi[k] for k in keys])

    # Drop any non-finite or near-zero Fisher (numerical)
    mask = np.isfinite(b) & np.isfinite(w) & np.isfinite(f) & (f > 0)
    b, w, f = b[mask], w[mask], f[mask]

    rho_bf = spearmanr(b, f).statistic
    rho_bw = spearmanr(b, w).statistic
    rho_fw = spearmanr(f, w).statistic
    rho_bf_w = partial_spearman(b, f, w)

    print(f'\nseed={seed}, task={args.task_id}, n_channels={len(b)}')
    print(f'  rho(b, F)        = {rho_bf:+.3f}')
    print(f'  rho(b, |w|)      = {rho_bw:+.3f}')
    print(f'  rho(F, |w|)      = {rho_fw:+.3f}')
    print(f'  rho(b, F | |w|)  = {rho_bf_w:+.3f}   <-- partial correlation')

    out = args.ckpt.replace('.pt', f'_partialrho_t{args.task_id}.json')
    with open(out, 'w') as fh:
        json.dump({
            'seed': seed, 'task_id': args.task_id, 'n_channels': int(len(b)),
            'rho_b_F': float(rho_bf), 'rho_b_w': float(rho_bw),
            'rho_F_w': float(rho_fw), 'rho_b_F_given_w': float(rho_bf_w),
        }, fh, indent=2)
    print(f'\nSaved: {out}')


if __name__ == '__main__':
    main()
