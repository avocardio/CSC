"""Unified supervised continual-learning runner.

One CLI, one script, all methods. Designed to match the RL CSC structure so
the paper has a coherent story across regimes.

Methods (selected via --method):
  finetune    : sequential CE, no protection (lower bound)
  replay      : experience replay, R samples/task
  der         : DER++ (Buzzega 2020), MSE on stored logits + CE on labels
  ewc         : Elastic Weight Consolidation (Kirkpatrick 2017)
  packnet     : prune-then-freeze (Mallya 2018) — magnitude-based ownership
  csc         : self-compression with quantization in forward; soft protection
                via per-channel gradient scaling by accumulated bit-depth.
                Optionally combine with DER++ via --der_alpha > 0.

Architecture: Quantized ResNet-18 with 32x32 stem, multi-head classifier.
Dataset: Split CIFAR-100 (--num_tasks splits 100 classes into equal groups).

CLI examples:
  python experiments/run_supervised.py --method csc --num_tasks 10 --gamma_comp 0.001
  python experiments/run_supervised.py --method der --replay_per_task 200
  python experiments/run_supervised.py --method packnet --prune 0.75 --retrain_epochs 10
"""
from __future__ import annotations
import os, sys, json, time, argparse, copy, math
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.resnet import (QuantizedResNet18, QuantizedResNet50,
                            QuantizedResNet101, QuantizedConv2d)
from models.convnext import convnext_tiny
from models.mlp import QuantizedMLP
from models.mlp import QuantizedLinear as QuantizedLinearMLP
from models.quantization import (
    DifferentiableQuantizer, get_quantizers, get_compression_stats,
    compute_average_bit_depth,
)
from training.metrics import evaluate_task, evaluate_all_tasks, CLMetrics
from data.split_cifar100 import SplitCIFAR100
from data.permuted_mnist import PermutedMNIST


def _is_quantized_module(m) -> bool:
    """A QuantizedConv2d or QuantizedLinear (the MLP one) — both expose .quantizer
    when `do_quantize=True` and have a `.weight` we can scale gradients on."""
    return isinstance(m, (QuantizedConv2d, QuantizedLinearMLP)) and getattr(m, 'do_quantize', False)


def _module_weight(m):
    """Underlying weight tensor (different attribute path for Conv vs Linear)."""
    if isinstance(m, QuantizedConv2d):
        return m.conv.weight
    return m.linear.weight


def _module_out_channels(m) -> int:
    """Number of output channels (Conv) or output features (Linear)."""
    if isinstance(m, QuantizedConv2d):
        return m.out_channels
    return m.linear.out_features

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


# ============================================================
# Replay / DER++ buffers (in-memory, GPU-resident)
# ============================================================
class ReplayBuffer:
    """Per-task slice of stored examples. Each slice has fixed size R."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.x: list[torch.Tensor] = []         # (Rs, 3, H, W) per task
        self.y: list[torch.Tensor] = []         # (Rs,) per task
        self.logits: list[torch.Tensor] = []    # (Rs, num_classes_per_task) — for DER

    @property
    def n_tasks(self) -> int:
        return len(self.x)

    def add_task(self, train_loader, model, task_id: int, n_per_task: int,
                 store_logits: bool = False):
        """Reservoir-style sample n_per_task examples from the just-finished task,
        then snapshot the model's logits on them (for DER++)."""
        xs, ys = [], []
        total = 0
        # Iterate dataset once and randomly subsample
        all_x, all_y = [], []
        for batch in train_loader:
            x, y = batch[0], batch[1]
            all_x.append(x); all_y.append(y)
        X = torch.cat(all_x); Y = torch.cat(all_y)
        n = min(n_per_task, X.shape[0])
        idx = torch.randperm(X.shape[0])[:n]
        x_keep = X[idx].to(self.device)
        y_keep = Y[idx].to(self.device)
        self.x.append(x_keep)
        self.y.append(y_keep)
        if store_logits:
            with torch.no_grad():
                model.eval()
                lg = model(x_keep, task_id=task_id)
                self.logits.append(lg.detach())
        else:
            self.logits.append(torch.zeros(0, device=self.device))

    def sample(self, batch_size: int):
        """Sample (x, y, t, logits_or_None) split across stored tasks."""
        if not self.x:
            return None
        T = len(self.x)
        per = max(1, batch_size // T)
        rem = batch_size - per * T
        out_x, out_y, out_t, out_lg = [], [], [], []
        for k in range(T):
            this_n = per + (1 if k < rem else 0)
            n_avail = self.x[k].shape[0]
            if this_n == 0 or n_avail == 0:
                continue
            idx = torch.randint(0, n_avail, (this_n,), device=self.device)
            out_x.append(self.x[k][idx])
            out_y.append(self.y[k][idx])
            out_t.append(torch.full((this_n,), k, dtype=torch.long, device=self.device))
            if self.logits[k].numel() > 0:
                out_lg.append(self.logits[k][idx])
        x = torch.cat(out_x); y = torch.cat(out_y); t = torch.cat(out_t)
        lg = torch.cat(out_lg) if out_lg else None
        return x, y, t, lg


# ============================================================
# Soft-protection (CSC)
# ============================================================
class SoftProtect:
    """Scale weight gradients by 1 / (1 + beta * acc_b) per output channel.
    Works for both QuantizedConv2d (4D weight) and QuantizedLinear (2D weight).

    `acc_b[layer]` is the running max of the layer's per-channel bit-depth across
    completed tasks. Updated by `on_task_end()`.
    """

    def __init__(self, model, beta: float = 1.0):
        self.beta = beta
        self.model = model
        self.acc_bits: dict[str, torch.Tensor] = {}
        for name, m in model.named_modules():
            if _is_quantized_module(m):
                self.acc_bits[name] = torch.zeros(_module_out_channels(m), device=DEVICE)

    @torch.no_grad()
    def on_task_end(self):
        for name, m in self.model.named_modules():
            if name not in self.acc_bits:
                continue
            b = m.quantizer.get_channel_bit_depths().detach().clamp(min=0)
            self.acc_bits[name] = torch.max(self.acc_bits[name], b)

    @torch.no_grad()
    def revive_low_bd_channels(self, threshold: float, init_bit_depth: float = 8.0,
                               init_exponent: float = -4.0) -> int:
        """Reset channels that are CURRENTLY low AND were NEVER important.

        Selectivity: a channel is revived only if both
          - bd_current < threshold  (low after this task), and
          - acc_bits   < threshold  (low across ALL prior tasks too)
        The acc_bits check protects channels that were specialists for any
        prior task — those keep their learned representations.
        Returns the total number of channels revived across the model."""
        import math
        revived = 0
        for name, m in self.model.named_modules():
            if name not in self.acc_bits:
                continue
            bd = m.quantizer.get_channel_bit_depths().detach().clamp(min=0)
            mask = (bd < threshold) & (self.acc_bits[name] < threshold)
            n = int(mask.sum().item())
            if n == 0:
                continue
            revived += n
            # Reset quantizer state on those channels
            m.quantizer.bit_depth.data[mask] = init_bit_depth
            m.quantizer.exponent.data[mask] = init_exponent
            # Reset weights for those channels (kaiming, matching upstream init).
            W = _module_weight(m)
            if W.dim() == 4:
                fan = W.shape[1] * W.shape[2] * W.shape[3]
                std = math.sqrt(2.0 / fan)
                W.data[mask] = torch.randn_like(W.data[mask]) * std
            else:
                fan = W.shape[1]
                bound = math.sqrt(1.0 / fan)
                W.data[mask] = torch.empty_like(W.data[mask]).uniform_(-bound, bound)
            # Reset acc_bits so soft-protect doesn't keep these channels semi-locked.
            self.acc_bits[name][mask] = 0.0
        return revived

    def scale_grads(self):
        if self.beta <= 0:
            return
        for name, m in self.model.named_modules():
            if name not in self.acc_bits:
                continue
            acc = self.acc_bits[name]
            scale = 1.0 / (1.0 + self.beta * acc)            # (out_channels,)
            W = _module_weight(m)
            if W.grad is None:
                continue
            if W.dim() == 4:
                W.grad.mul_(scale.view(-1, 1, 1, 1))         # Conv2d (O,I,kH,kW)
            else:
                W.grad.mul_(scale.view(-1, 1))               # Linear (O, I)


def control_as_fisher(soft_protect: 'SoftProtect', ewc: 'EWCRegularizer',
                      kind: str, seed: int = 0):
    """csc_*_ewc control variants: P5.1 ablation.
    Replace the empirical Fisher with one of:
       'random'    — per-channel random importance ∈ [0,1] (seeded reproducibly).
       'magnitude' — per-channel max|weight| (post-training).
    Same structure as bd_as_fisher: snapshot params, broadcast over fan-in."""
    import torch as _torch
    g = _torch.Generator(device=DEVICE).manual_seed(seed)
    ewc._snapshot_params()
    name_to_module = {n: m for n, m in soft_protect.model.named_modules()}
    for mod_name, acc in soft_protect.acc_bits.items():
        m = name_to_module.get(mod_name)
        if m is None:
            continue
        W = _module_weight(m)
        for pname, p in soft_protect.model.named_parameters():
            if p is W:
                break
        else:
            continue
        if kind == 'random':
            sig = _torch.rand(acc.shape, generator=g, device=acc.device).clamp(min=1e-8)
        elif kind == 'magnitude':
            flat = W.detach().reshape(W.shape[0], -1).abs()
            sig = flat.amax(dim=1).clamp(min=1e-8)
        else:
            raise ValueError(kind)
        if W.dim() == 4:
            f = sig.view(-1, 1, 1, 1).expand_as(W).contiguous()
        else:
            f = sig.view(-1, 1).expand_as(W).contiguous()
        ewc.fisher[pname] = ewc.fisher.get(pname, _torch.zeros_like(f)) + f


def bd_as_fisher(soft_protect: 'SoftProtect', ewc: 'EWCRegularizer'):
    """csc_bd_ewc helper: copy CSC's accumulated per-channel bit-depths into
    EWCRegularizer.fisher (broadcast over fan-in), and snapshot params. Used in
    place of EWCRegularizer.on_task_end() when --method=csc_bd_ewc — tests
    whether the bit-depth signal serves as a Fisher proxy."""
    ewc._snapshot_params()
    name_to_module = {n: m for n, m in soft_protect.model.named_modules()}
    for mod_name, acc in soft_protect.acc_bits.items():
        m = name_to_module.get(mod_name)
        if m is None:
            continue
        W = _module_weight(m)
        # Find the parameter name for this weight (need to match model.named_parameters key)
        for pname, p in soft_protect.model.named_parameters():
            if p is W:
                break
        else:
            continue
        bd = acc.detach().clamp(min=1e-8)                # avoid 0 importance
        if W.dim() == 4:
            f = bd.view(-1, 1, 1, 1).expand_as(W).contiguous()
        else:
            f = bd.view(-1, 1).expand_as(W).contiguous()
        ewc.fisher[pname] = ewc.fisher.get(pname, torch.zeros_like(f)) + f


# ============================================================
# EWC (light, supervised version — empirical Fisher)
# ============================================================
class EWCRegularizer:
    """Empirical Fisher diag from per-sample log-prob gradients (vmap+grad)."""

    def __init__(self, model, lam: float = 1000.0):
        self.lam = lam
        self.model = model
        self.fisher: dict[str, torch.Tensor] = {}
        self.params: dict[str, torch.Tensor] = {}

    def penalty(self) -> torch.Tensor:
        if not self.fisher:
            return torch.zeros((), device=next(self.model.parameters()).device)
        loss = 0.0
        for n, p in self.model.named_parameters():
            if n in self.fisher:
                loss = loss + (self.fisher[n] * (p - self.params[n]).pow(2)).sum()
        return self.lam * loss

    @torch.no_grad()
    def _snapshot_params(self):
        for n, p in self.model.named_parameters():
            if p.requires_grad:
                self.params[n] = p.detach().clone()

    def on_task_end(self, train_loader, task_id: int, n_batches: int = 32):
        from torch.func import functional_call, grad, vmap
        self._snapshot_params()
        params = {n: p.detach() for n, p in self.model.named_parameters()}
        names = list(params.keys())
        device = next(self.model.parameters()).device
        F_acc = {n: torch.zeros_like(params[n]) for n in names}

        # Eval mode disables BN's num_batches_tracked.add_ which cannot run under
        # vmap/grad. Restored after the Fisher pass.
        was_training = self.model.training
        self.model.eval()

        def f_logp(p_dict, x_one, y_one):
            x = x_one.unsqueeze(0)
            logits = functional_call(self.model, p_dict, args=(x,),
                                     kwargs={'task_id': task_id})
            return -F.cross_entropy(logits, y_one.unsqueeze(0))

        n_done = 0
        for k, batch in enumerate(train_loader):
            if k >= n_batches:
                break
            x = batch[0].to(device); y = batch[1].to(device)
            g = vmap(grad(f_logp), in_dims=(None, 0, 0))(params, x, y)
            for n in names:
                if n in g:
                    F_acc[n] = F_acc[n] + g[n].pow(2).sum(0)
            n_done += x.shape[0]
        for n in names:
            f = (F_acc[n] / max(n_done, 1)).clamp(min=1e-8)
            self.fisher[n] = self.fisher.get(n, torch.zeros_like(f)) + f

        if was_training:
            self.model.train()


# ============================================================
# PackNet
# ============================================================
class PackNet:
    def __init__(self, model: QuantizedResNet18, prune: float = 0.75,
                 retrain_epochs: int = 5, lr_retrain: float = 1e-3):
        self.model = model
        self.prune = prune
        self.retrain_epochs = retrain_epochs
        self.lr_retrain = lr_retrain
        # Per-conv ownership (uint8 tensor; 0 = current free, 1..T = owned by that task)
        self.owner: dict[str, torch.Tensor] = {}
        for name, m in model.named_modules():
            if isinstance(m, QuantizedConv2d):
                self.owner[name] = torch.zeros_like(m.conv.weight, dtype=torch.uint8)

    def mask_grads(self, current_task: int):
        for name, m in self.model.named_modules():
            if name in self.owner and m.conv.weight.grad is not None:
                # Only weights owned by current_task (or free, marked 0) get gradient
                allow = (self.owner[name] == 0) | (self.owner[name] == current_task + 1)
                m.conv.weight.grad.mul_(allow.float())

    @torch.no_grad()
    def restore_frozen(self, snapshot: dict[str, torch.Tensor], current_task: int):
        for name, m in self.model.named_modules():
            if name in snapshot:
                frozen = (self.owner[name] != 0) & (self.owner[name] != current_task + 1)
                m.conv.weight.data = torch.where(frozen, snapshot[name],
                                                 m.conv.weight.data)

    @torch.no_grad()
    def snapshot(self) -> dict[str, torch.Tensor]:
        return {name: m.conv.weight.data.clone()
                for name, m in self.model.named_modules() if name in self.owner}

    @torch.no_grad()
    def prune_after_task(self, current_task: int):
        """Of weights still 'free' (owner==0), keep top (1-prune) by magnitude
        for this task (mark them owned by current_task+1); zero the rest."""
        for name, m in self.model.named_modules():
            if name not in self.owner:
                continue
            owner = self.owner[name]
            free = (owner == 0)
            W = m.conv.weight.data
            if free.sum().item() == 0:
                continue
            free_vals = W[free].abs()
            k = max(1, int(free_vals.numel() * (1 - self.prune)))
            threshold = free_vals.kthvalue(free_vals.numel() - k + 1).values.item()
            keep = free & (W.abs() >= threshold)
            zero = free & ~keep
            W[zero] = 0.0
            owner[keep] = current_task + 1


# ============================================================
# Train one task
# ============================================================
def train_one_task(model, train_loader, task_id: int, n_epochs: int, lr: float,
                   replay: ReplayBuffer | None, replay_ratio: float,
                   der_alpha: float, der_beta: float,
                   csc_gamma: float, soft: SoftProtect | None,
                   ewc: EWCRegularizer | None, packnet: PackNet | None,
                   bias_l1: float, weight_decay: float):
    device = next(model.parameters()).device
    use_cuda = device.type == 'cuda'
    is_csc = csc_gamma > 0

    # Two parameter groups: weight params + (if csc) quantizer params (different LR)
    quant_params, weight_params = [], []
    for m in model.modules():
        if isinstance(m, DifferentiableQuantizer):
            quant_params.extend(list(m.parameters()))
    qids = {id(p) for p in quant_params}
    for p in model.parameters():
        if id(p) not in qids:
            weight_params.append(p)

    pg = [{'params': weight_params, 'lr': lr, 'weight_decay': weight_decay}]
    if is_csc and quant_params:
        pg.append({'params': quant_params, 'lr': 0.5, 'weight_decay': 0.0, 'eps': 1e-3})
    optimizer = torch.optim.AdamW(pg)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=n_epochs * max(1, len(train_loader)))
    scaler = torch.amp.GradScaler('cuda', enabled=use_cuda)

    pn_snapshot = packnet.snapshot() if packnet else None

    n_iter = 0
    for epoch in range(n_epochs):
        model.train()
        running = {'task': 0.0, 'comp': 0.0, 'replay': 0.0, 'n': 0}
        for batch in train_loader:
            x, y = batch[0].to(device, non_blocking=True), batch[1].to(device, non_blocking=True)
            with torch.amp.autocast('cuda', enabled=use_cuda):
                logits = model(x, task_id=task_id)
                loss = F.cross_entropy(logits, y)
                running['task'] += loss.item()

                # Replay / DER++
                if replay is not None and replay.n_tasks > 0:
                    rep = replay.sample(int(x.shape[0] * replay_ratio) or 1)
                    if rep is not None:
                        rx, ry, rt, rlg = rep
                        rep_loss = 0.0
                        # Group by task to apply correct head
                        for tid in rt.unique():
                            mask = rt == tid
                            r_logits = model(rx[mask], task_id=int(tid.item()))
                            rep_ce = F.cross_entropy(r_logits, ry[mask])
                            rep_loss = rep_loss + rep_ce
                            if rlg is not None and der_alpha > 0:
                                rep_loss = rep_loss + der_alpha * F.mse_loss(
                                    r_logits, rlg[mask])
                        rep_loss = rep_loss / max(rt.unique().numel(), 1)
                        loss = loss + der_beta * rep_loss
                        running['replay'] += rep_loss.item()

                # Compression objective
                if is_csc:
                    Q = compute_average_bit_depth(model, use_coupled=False)
                    loss = loss + csc_gamma * Q
                    running['comp'] += float(Q.detach().item())

                # Bias L1 for zeroed channels (small regularizer)
                if bias_l1 > 0:
                    bl = sum(m.bias.abs().mean() for m in model.modules()
                             if isinstance(m, QuantizedConv2d) and m.conv.bias is not None)
                    if isinstance(bl, torch.Tensor):
                        loss = loss + bias_l1 * bl

                # EWC penalty
                if ewc is not None:
                    loss = loss + ewc.penalty()

            optimizer.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            if soft is not None:
                soft.scale_grads()
            if packnet is not None:
                packnet.mask_grads(task_id)

            scaler.step(optimizer)
            scaler.update()
            scheduler.step()

            if packnet is not None:
                packnet.restore_frozen(pn_snapshot, task_id)
            running['n'] += 1
            n_iter += 1
        if (epoch + 1) % 5 == 0 or epoch == 0:
            n = running['n']
            stats = get_compression_stats(model) if is_csc else None
            tag = '' if not is_csc else f' bits={stats["compression_ratio"]*100:.1f}%'
            print(f'    epoch {epoch+1:3d}/{n_epochs} '
                  f'task_loss={running["task"]/n:.3f} '
                  f'replay={running["replay"]/max(n,1):.3f}{tag}',
                  flush=True)


# ============================================================
# Main entry point
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True,
                   choices=['finetune', 'replay', 'der', 'ewc', 'packnet', 'csc',
                            'csc_ewc', 'csc_bd_ewc', 'csc_bd_ewc_warm',
                            'csc_random_ewc', 'csc_magnitude_ewc'])
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--epochs_per_task', type=int, default=50)
    p.add_argument('--batch_size', type=int, default=128)
    p.add_argument('--lr', type=float, default=1e-3)
    p.add_argument('--weight_decay', type=float, default=5e-4)
    p.add_argument('--seed', type=int, default=42)
    # Replay / DER
    p.add_argument('--replay_per_task', type=int, default=200)
    p.add_argument('--replay_ratio', type=float, default=0.5)
    p.add_argument('--der_alpha', type=float, default=0.0,
                   help='MSE-on-logits weight (DER++); 0 for ER only')
    p.add_argument('--der_beta', type=float, default=1.0,
                   help='Multiplier on replay loss')
    # CSC
    p.add_argument('--gamma_comp', type=float, default=0.001)
    p.add_argument('--soft_beta', type=float, default=1.0)
    p.add_argument('--bias_l1', type=float, default=0.0)
    # EWC
    p.add_argument('--ewc_lambda', type=float, default=1000.0)
    p.add_argument('--ewc_n_batches', type=int, default=20,
                   help='Batches used for empirical Fisher pass (default 20).')
    p.add_argument('--bd_revive_threshold', type=float, default=0.0,
                   help='If > 0, channels with bit-depth below this threshold are '
                        're-initialised at task end (free capacity for next task).')
    p.add_argument('--bd_init_scale', type=float, default=1e-3,
                   help='Scale factor when initialising EWC fisher from bit-depth '
                        '(csc_bd_ewc_warm). Brings bd values close to typical '
                        'empirical Fisher magnitude so they compose meaningfully.')
    # PackNet
    p.add_argument('--prune', type=float, default=0.75)
    p.add_argument('--retrain_epochs', type=int, default=10)
    # Dataset / model
    p.add_argument('--dataset', default='cifar100',
                   choices=['cifar100', 'pmnist'],
                   help='Split CIFAR-100 (multi-head ResNet) or Permuted MNIST (single-head MLP)')
    p.add_argument('--model', default='resnet18',
                   choices=['resnet18', 'resnet50', 'resnet101',
                            'convnext_tiny', 'mlp'])
    p.add_argument('--data_root', default='/mnt/e/datasets/cifar100')
    p.add_argument('--tag', default='')
    p.add_argument('--save_model_final', action='store_true',
                   help='Save final model state_dict for post-hoc analysis (PTQ etc.)')
    args = p.parse_args()

    torch.manual_seed(args.seed); np.random.seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)
        torch.backends.cudnn.benchmark = True

    device = DEVICE
    quantize = args.method in ('csc', 'csc_ewc', 'csc_bd_ewc', 'csc_bd_ewc_warm',
                               'csc_random_ewc', 'csc_magnitude_ewc')

    # Dataset + model factory
    if args.dataset == 'cifar100':
        classes_per_task = 100 // args.num_tasks
        benchmark = SplitCIFAR100(data_root=args.data_root,
                                  num_tasks=args.num_tasks,
                                  batch_size=args.batch_size,
                                  seed=args.seed)
        if args.model == 'resnet50':
            model = QuantizedResNet50(num_classes_per_task=classes_per_task,
                                      num_tasks=args.num_tasks,
                                      quantize=quantize).to(device)
        elif args.model == 'resnet101':
            model = QuantizedResNet101(num_classes_per_task=classes_per_task,
                                       num_tasks=args.num_tasks,
                                       quantize=quantize).to(device)
        elif args.model == 'convnext_tiny':
            model = convnext_tiny(num_classes_per_task=classes_per_task,
                                  num_tasks=args.num_tasks,
                                  quantize=quantize).to(device)
        else:
            model = QuantizedResNet18(num_classes_per_task=classes_per_task,
                                      num_tasks=args.num_tasks,
                                      quantize=quantize).to(device)
    elif args.dataset == 'pmnist':
        # Single-head MLP, all tasks share the 10-class output.
        # MNIST default path; allow override via --data_root.
        if args.data_root.endswith('cifar100'):                            # auto-default if not changed
            args.data_root = '/mnt/e/datasets/mnist'
        benchmark = PermutedMNIST(data_root=args.data_root,
                                  num_tasks=args.num_tasks,
                                  batch_size=args.batch_size,
                                  seed=args.seed)
        model = QuantizedMLP(input_size=784, hidden_size=256, num_classes=10,
                             num_tasks=1,                                  # single head
                             quantize=quantize).to(device)
    else:
        raise ValueError(f'Unknown dataset {args.dataset}')

    print(f'Method={args.method} dataset={args.dataset} num_tasks={args.num_tasks} '
          f'epochs/task={args.epochs_per_task} seed={args.seed}',
          flush=True)

    use_replay = args.method in ('replay', 'der', 'csc') and args.replay_per_task > 0
    use_der = args.method == 'der' or (args.method == 'csc' and args.der_alpha > 0)
    # csc_ewc: vanilla EWC penalty + CSC compression + soft-protect, no replay/DER.
    # csc_bd_ewc: same, but the EWC penalty's importance weight is the per-channel
    #             bit-depth (broadcast over fan-in) instead of the empirical Fisher.
    # csc_bd_ewc_warm: bd initialises EWC's fisher (scaled), then refine with K
    #                  batches of empirical Fisher on top — quantifies how cheap
    #                  Fisher refinement gets when warm-started from bd.
    use_ewc = args.method in ('ewc', 'csc_ewc', 'csc_bd_ewc', 'csc_bd_ewc_warm',
                              'csc_random_ewc', 'csc_magnitude_ewc')
    use_pn = args.method == 'packnet'
    use_csc = args.method in ('csc', 'csc_ewc', 'csc_bd_ewc', 'csc_bd_ewc_warm',
                              'csc_random_ewc', 'csc_magnitude_ewc')
    use_bd_ewc = args.method == 'csc_bd_ewc'
    use_bd_ewc_warm = args.method == 'csc_bd_ewc_warm'
    use_random_ewc = args.method == 'csc_random_ewc'
    use_magnitude_ewc = args.method == 'csc_magnitude_ewc'

    replay = ReplayBuffer(device) if use_replay else None
    soft = SoftProtect(model, beta=args.soft_beta) if use_csc else None
    ewc = EWCRegularizer(model, lam=args.ewc_lambda) if use_ewc else None
    packnet = PackNet(model, prune=args.prune,
                      retrain_epochs=args.retrain_epochs) if use_pn else None
    cl_metrics = CLMetrics(args.num_tasks)
    t0 = time.time()
    # Per-task bit-depth trajectories: list of {layer_name: list[float]} per task end.
    # Captures the actual end-of-task bit-depth (not the running max in soft.acc_bits)
    # so we can plot trajectories — does any channel rise specifically on task k,
    # then get reused (rise again differently) on task k+1?
    bd_trajectories: list[dict] = []

    for task_id in range(args.num_tasks):
        print(f'\n{"=" * 50}\nTASK {task_id}\n{"=" * 50}', flush=True)
        train_loader, _ = benchmark.get_task_dataloaders(task_id)

        train_one_task(
            model, train_loader, task_id, args.epochs_per_task, args.lr,
            replay if use_replay else None, args.replay_ratio,
            args.der_alpha if use_der else 0.0, args.der_beta,
            args.gamma_comp if use_csc else 0.0, soft, ewc, packnet,
            args.bias_l1, args.weight_decay)

        # PackNet: prune then retrain on free weights only
        if use_pn:
            packnet.prune_after_task(task_id)
            print(f'    PackNet: pruned {args.prune*100:.0f}%, retraining...', flush=True)
            train_one_task(model, train_loader, task_id, args.retrain_epochs,
                           args.lr, None, 0, 0, 0, 0, None, None, packnet, 0,
                           args.weight_decay)

        # Snapshot replay buffer with current model logits (for DER++)
        if use_replay:
            replay.add_task(train_loader, model, task_id, args.replay_per_task,
                            store_logits=use_der)
        if use_csc:
            soft.on_task_end()
            # Snapshot per-channel bit-depth at end of this task.
            snap = {}
            for name, m in model.named_modules():
                if _is_quantized_module(m):
                    snap[name] = m.quantizer.get_channel_bit_depths().detach().clamp(min=0).cpu().tolist()
            bd_trajectories.append(snap)
            # Revive low-bit-depth channels so next task can use them.
            if args.bd_revive_threshold > 0 and task_id < args.num_tasks - 1:
                n = soft.revive_low_bd_channels(args.bd_revive_threshold)
                print(f'  CSC: revived {n} channels (bd < {args.bd_revive_threshold})', flush=True)
        if use_random_ewc:
            control_as_fisher(soft, ewc, 'random', seed=args.seed + task_id)
        elif use_magnitude_ewc:
            control_as_fisher(soft, ewc, 'magnitude')
        elif use_bd_ewc:
            # bd as Fisher proxy, no real Fisher pass.
            bd_as_fisher(soft, ewc)
        elif use_bd_ewc_warm:
            # Warm-start: scale bd to ~Fisher magnitude, then refine with K batches.
            # bd values are O(1)-O(10); Fisher values are typically O(1e-4)-O(1e-3),
            # so we multiply bd by args.bd_init_scale before storing in ewc.fisher.
            bd_as_fisher(soft, ewc)
            # Rescale the bd-init component so it doesn't dominate the refinement.
            for n in list(ewc.fisher.keys()):
                ewc.fisher[n] = ewc.fisher[n] * args.bd_init_scale
            ewc.on_task_end(train_loader, task_id, n_batches=args.ewc_n_batches)
        elif use_ewc:
            ewc.on_task_end(train_loader, task_id, n_batches=args.ewc_n_batches)

        accs = evaluate_all_tasks(model, benchmark, task_id + 1, device)
        cl_metrics.update(task_id, accs)
        for j, a in enumerate(accs):
            print(f'  Task {j}: {a*100:.1f}%')
        print(f'  AvgSeen: {cl_metrics.average_accuracy(task_id)*100:.2f}%', flush=True)

    cl_metrics.print_matrix()
    if use_csc:
        stats = get_compression_stats(model)
        print(f'  Compression bits: {stats["compression_ratio"]*100:.2f}% '
              f'({stats["zero_channels"]}/{stats["total_channels"]} dead channels)',
              flush=True)

    elapsed = time.time() - t0
    print(f'  Wall time: {elapsed:.0f}s', flush=True)

    # Model name in filename only when not the default — keeps existing
    # CIFAR-100 ResNet-18 JSON paths backward-compatible.
    model_part = '' if args.model == 'resnet18' else f'_{args.model}'
    out_path = (f'checkpoints/sup_{args.dataset}_{args.method}'
                f'{model_part}_t{args.num_tasks}_s{args.seed}{args.tag}.json')
    os.makedirs('checkpoints', exist_ok=True)
    out = {
        'config': vars(args),
        'final_avg': cl_metrics.final_average(),
        'bwt': cl_metrics.backward_transfer(),
        'forgetting': cl_metrics.forgetting(),
        'accuracy_matrix': cl_metrics.accuracy_matrix,
        'wall_time': elapsed,
    }
    if use_csc:
        out['compression'] = stats
        out['bd_trajectories'] = bd_trajectories
    with open(out_path, 'w') as f:
        json.dump(out, f, indent=2)
    print(f'Saved: {out_path}', flush=True)

    if args.save_model_final:
        pt_path = out_path.replace('.json', '_final.pt')
        torch.save({'model': model.state_dict(),
                    'classes_per_task': classes_per_task,
                    'num_tasks': args.num_tasks,
                    'seed': args.seed,
                    'method': args.method,
                    'model_arch': args.model,
                    'dataset': args.dataset}, pt_path)
        print(f'Saved model: {pt_path}', flush=True)


if __name__ == '__main__':
    main()
