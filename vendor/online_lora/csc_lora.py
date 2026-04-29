"""CSC integration for Online-LoRA. Modular hooks; minimal entanglement with
upstream engine.py / lora.py.

What this module provides:

  * `wrap_lora_with_csc(lora_model)` — replaces wnew_a / wnew_b Linear weights
    with a CSC-style differentiable quantizer (per-channel learnable bit-depth
    and exponent, STE-rounded forward). Other modules untouched.

  * `lora_compression_loss(lora_model)` — gamma * average-bit-depth across all
    quantized wnew layers (the standard CSC objective term).

  * `lora_bd_omegas(lora_model)` — returns numpy arrays shaped exactly like
    upstream `omega_As` / `omega_Bs`, populated from the per-channel
    accumulated bit-depth (broadcast over fan-in) instead of MAS gradient²
    on the 4-sample hard buffer.

  * `lora_acc_bits_update(lora_model)` — accumulate per-channel max bit-depth
    across importance updates (analogue of Online-LoRA's running-mean omega).

The upstream engine remains the source of truth for the training loop, drift
detector, and overall logic. We swap exactly two arrays at one call site.
"""
from __future__ import annotations
import os, sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# Use the project-level CSC quantizer so the implementation stays single-source.
# vendor/online_lora/ -> continual_learning/  (parent of vendor)
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_THIS_DIR, '..', '..'))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from models.quantization import (
    DifferentiableQuantizer, CompressionGranularity,
)


class _QuantizedLoRALinear(nn.Module):
    """Drop-in replacement for an nn.Linear inside Online-LoRA's wnew_* slot.

    Forward path is exactly Linear(x, q(W)). Bias is None (matches upstream
    LoRA which never uses bias on wnew layers). The per-channel quantizer is
    accessed as `.quantizer` for the CSC compression objective and bd-omega
    extraction.
    """

    def __init__(self, in_features: int, out_features: int,
                 init_bit_depth: float = 8.0, init_exponent: float = -4.0):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.weight, a=5 ** 0.5)
        self.quantizer = DifferentiableQuantizer(
            self.weight.shape,
            granularity=CompressionGranularity.CHANNEL,
            init_bit_depth=init_bit_depth,
            init_exponent=init_exponent,
        )
        # Track the running max bit-depth across importance updates so the
        # bd-omega is monotone-non-decreasing across the lifetime of the
        # current LoRA generation (matches the spirit of upstream's
        # 1/count_updates EMA where omega only grows).
        self.register_buffer('acc_bits',
                             torch.zeros(out_features), persistent=True)

    def forward(self, x):
        qw = self.quantizer(self.weight)
        return F.linear(x, qw)

    @torch.no_grad()
    def update_acc_bits(self):
        b = self.quantizer.get_channel_bit_depths().detach().clamp(min=0)
        self.acc_bits = torch.maximum(self.acc_bits, b)


def wrap_lora_with_csc(lora_model, init_bit_depth: float = 8.0):
    """Replace each wnew_a* / wnew_b* Linear in lora_model.wnew_As / wnew_Bs with
    `_QuantizedLoRALinear`. The replacement preserves the `_is_wnew_a` /
    `_is_wnew_b` parameter flags upstream code uses to filter parameters.

    Mutates `lora_model` in-place. Should be called immediately after
    `LoRA_ViT_timm(...)` construction, before the optimizer is built.

    Returns the list of installed _QuantizedLoRALinear modules (in (a,b) pairs)
    for convenient access from the engine.
    """
    qlist = []
    # The lora_model exposes wnew_As, wnew_Bs as parallel lists. Each entry is
    # the Linear module that the actual qkv layer holds a *reference* to. We
    # therefore have to also patch the references inside each block.
    # In their _LoRA_qkv_timm they store wnew_a_linear_q, wnew_b_linear_q,
    # wnew_a_linear_v, wnew_b_linear_v as attributes. We swap those.
    n = len(lora_model.wnew_As)
    new_wnew_As = []
    new_wnew_Bs = []
    for i in range(n):
        old_a = lora_model.wnew_As[i]
        old_b = lora_model.wnew_Bs[i]
        qa = _QuantizedLoRALinear(old_a.in_features, old_a.out_features,
                                  init_bit_depth=init_bit_depth)
        qb = _QuantizedLoRALinear(old_b.in_features, old_b.out_features,
                                  init_bit_depth=init_bit_depth)
        # Preserve the upstream "_is_wnew_a/_b" markers on .weight.
        setattr(qa.weight, '_is_wnew_a', True)
        setattr(qb.weight, '_is_wnew_b', True)
        new_wnew_As.append(qa)
        new_wnew_Bs.append(qb)
        qlist.append((qa, qb))

    # Rebuild lora_model.wnew_As / wnew_Bs with the new modules.
    lora_model.wnew_As = nn.ModuleList(new_wnew_As)
    lora_model.wnew_Bs = nn.ModuleList(new_wnew_Bs)

    # Now patch each qkv block's references. The block's qkv module holds
    # attributes named wnew_a_linear_q, wnew_b_linear_q, wnew_a_linear_v,
    # wnew_b_linear_v. We fed these in pairs (q-pair, v-pair) when constructing
    # the model: wnew_As[2k] = a_q, wnew_As[2k+1] = a_v, wnew_Bs[2k] = b_q,
    # wnew_Bs[2k+1] = b_v.
    blocks = list(lora_model.lora_vit.blocks)
    pair = 0
    for blk in blocks:
        qkv = blk.attn.qkv
        if not hasattr(qkv, 'wnew_a_linear_q'):
            continue
        qkv.wnew_a_linear_q = new_wnew_As[2 * pair]
        qkv.wnew_a_linear_v = new_wnew_As[2 * pair + 1]
        qkv.wnew_b_linear_q = new_wnew_Bs[2 * pair]
        qkv.wnew_b_linear_v = new_wnew_Bs[2 * pair + 1]
        pair += 1

    return qlist


def lora_compression_loss(lora_model) -> torch.Tensor:
    """Sum of average bit-depths across all quantized wnew layers. Multiply
    by gamma in the engine to get the CSC compression objective term."""
    losses = []
    device = next(lora_model.parameters()).device
    for m in list(lora_model.wnew_As) + list(lora_model.wnew_Bs):
        if isinstance(m, _QuantizedLoRALinear):
            losses.append(m.quantizer.get_bit_depths().clamp(min=0).mean())
    if not losses:
        return torch.zeros((), device=device)
    return sum(losses) / len(losses)


@torch.no_grad()
def lora_acc_bits_update(lora_model):
    for m in list(lora_model.wnew_As) + list(lora_model.wnew_Bs):
        if isinstance(m, _QuantizedLoRALinear):
            m.update_acc_bits()


@torch.no_grad()
def lora_reset_acc_bits(lora_model):
    """Called after `update_and_reset_lora_parameters` — the new generation's
    bit-depth accumulator starts fresh, and the quantizer state (bit_depth,
    exponent) is reset to init so the new generation re-learns its importance
    rather than inheriting the previous generation's bit-depth distribution."""
    for m in list(lora_model.wnew_As) + list(lora_model.wnew_Bs):
        if isinstance(m, _QuantizedLoRALinear):
            m.acc_bits.zero_()
            m.quantizer.bit_depth.data.fill_(m.quantizer.init_bit_depth)
            # Init exponent unspecified at quantizer-creation time defaults to -4.
            m.quantizer.exponent.data.fill_(-4.0)


def lora_bd_omegas(lora_model, init_bit_depth: float = 8.0):
    """Return (omega_As, omega_Bs) numpy lists with the same shape upstream
    expects (broadcast over fan-in), populated from accumulated per-channel
    bit-depths normalised to [0, 1].

    This is the drop-in replacement for the 4-sample MAS gradient² computation
    in engine.py's importance update branch.
    """
    omega_As, omega_Bs = [], []
    for ma, mb in zip(lora_model.wnew_As, lora_model.wnew_Bs):
        if not isinstance(ma, _QuantizedLoRALinear):
            # Fallback to zeros if not quantized (shouldn't happen if --csc set).
            omega_As.append(np.zeros(ma.weight.shape, dtype=np.float32))
            omega_Bs.append(np.zeros(mb.weight.shape, dtype=np.float32))
            continue
        bd_a = (ma.acc_bits.detach().cpu().numpy() / init_bit_depth).astype(np.float32)
        bd_b = (mb.acc_bits.detach().cpu().numpy() / init_bit_depth).astype(np.float32)
        # Broadcast (out,) over (out, in) to match Linear weight shape.
        wa = np.broadcast_to(bd_a[:, None], ma.weight.shape).copy()
        wb = np.broadcast_to(bd_b[:, None], mb.weight.shape).copy()
        omega_As.append(wa)
        omega_Bs.append(wb)
    return omega_As, omega_Bs
