"""Vision Transformer (ViT) for the model-size scaling axis.

Default ViT-Tiny (~5M) and ViT-Small (~22M) variants. Optimised path:

  - F.scaled_dot_product_attention auto-dispatches to FlashAttention on
    SM80+ (A100, H100, GH200) and other fused kernels otherwise. We don't
    write a custom attention.

  - Quantisation principle (consistent with the rest of the codebase):
    quantise every CORE weight matrix and skip only (a) the input
    embedding/stem and (b) the per-task output heads. For ViT this means
    QKV, attention output projection, and both MLP linears in every block
    are quantised; patch_embed (RGB -> tokens) and the per-task
    classification heads are not. This matches the supervised paper of
    Csefalvay & Imber and avoids two known failure modes: quantising the
    input transform destabilises from-scratch training, and quantising
    per-task heads conflicts with multi-head ownership in CL.

  - For 32x32 inputs we use patch_size=4 (so 8x8=64 patch tokens + 1 cls).

ViT-Tiny: 12 blocks x 4 quantised linears = 48 quantisers.
ViT-Small: same, with larger dim.
"""
from __future__ import annotations
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .mlp import QuantizedLinear


class _MLP(nn.Module):
    """Transformer MLP block: dim -> 4*dim -> dim, GELU. Both linears quantised."""
    def __init__(self, dim: int, hidden_dim: int, quantize: bool = True,
                 dropout: float = 0.0):
        super().__init__()
        self.fc1 = QuantizedLinear(dim, hidden_dim, quantize=quantize)
        self.fc2 = QuantizedLinear(hidden_dim, dim, quantize=quantize)
        self.drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

    def forward(self, x):
        return self.drop(self.fc2(F.gelu(self.fc1(x))))


class _Attention(nn.Module):
    """Multi-head self-attention using F.scaled_dot_product_attention.
    On Ampere+ this dispatches to FlashAttention automatically.
    qkv and proj linears are quantised when quantize=True (consistent with
    the principle: quantise every core weight matrix except input/output)."""
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True,
                 attn_drop: float = 0.0, proj_drop: float = 0.0,
                 quantize: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f'dim {dim} must be divisible by num_heads {num_heads}'
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        # qkv must keep bias; QuantizedLinear's underlying Linear has bias by default
        self.qkv = QuantizedLinear(dim, dim * 3, quantize=quantize)
        self.proj = QuantizedLinear(dim, dim, quantize=quantize)
        self.attn_drop_p = attn_drop
        self.proj_drop = nn.Dropout(proj_drop) if proj_drop > 0 else nn.Identity()

    def forward(self, x):
        B, N, C = x.shape
        qkv = (self.qkv(x)
               .reshape(B, N, 3, self.num_heads, self.head_dim)
               .permute(2, 0, 3, 1, 4))                          # (3, B, H, N, D)
        q, k, v = qkv.unbind(0)
        # FlashAttention path on supported hardware
        out = F.scaled_dot_product_attention(
            q, k, v, dropout_p=self.attn_drop_p if self.training else 0.0)
        out = out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(out))


class _Block(nn.Module):
    """Pre-norm transformer block: x -> x + Attn(LN(x)) -> x + MLP(LN(x))."""
    def __init__(self, dim: int, num_heads: int, mlp_ratio: float = 4.0,
                 quantize: bool = True, dropout: float = 0.0,
                 attn_drop: float = 0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = _Attention(dim, num_heads, attn_drop=attn_drop, quantize=quantize)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = _MLP(dim, int(dim * mlp_ratio), quantize=quantize, dropout=dropout)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class QuantizedViT(nn.Module):
    """ViT with channel-level differentiable quantisation on MLP linears.
    Multi-head classifier (one per task)."""

    def __init__(self, img_size: int = 32, patch_size: int = 4, in_channels: int = 3,
                 dim: int = 192, depth: int = 12, num_heads: int = 3,
                 mlp_ratio: float = 4.0, num_classes_per_task: int = 10,
                 num_tasks: int = 10, dropout: float = 0.0, attn_drop: float = 0.0,
                 quantize: bool = True):
        super().__init__()
        assert img_size % patch_size == 0
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.dim = dim
        self.depth = depth
        self.num_heads = num_heads
        self.num_tasks = num_tasks
        self.num_classes_per_task = num_classes_per_task

        self.patch_embed = nn.Conv2d(in_channels, dim, kernel_size=patch_size,
                                     stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, self.num_patches + 1, dim))
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        self.pos_drop = nn.Dropout(dropout) if dropout > 0 else nn.Identity()

        self.blocks = nn.ModuleList([
            _Block(dim, num_heads, mlp_ratio, quantize=quantize,
                   dropout=dropout, attn_drop=attn_drop)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(dim)

        # Per-task classification heads
        self.heads = nn.ModuleList([
            nn.Linear(dim, num_classes_per_task) for _ in range(num_tasks)
        ])
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Conv2d):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, x, task_id=None):
        B = x.shape[0]
        x = self.patch_embed(x).flatten(2).transpose(1, 2)        # (B, N, dim)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls, x], dim=1)
        x = self.pos_drop(x + self.pos_embed)
        for blk in self.blocks:
            x = blk(x)
        x = self.norm(x)
        cls_feat = x[:, 0]                                         # CLS token
        if task_id is not None:
            return self.heads[task_id](cls_feat)
        return torch.cat([h(cls_feat) for h in self.heads], dim=1)


# ============================================================
# Standard ViT presets (default sizes from the original ViT paper)
# ============================================================
def vit_tiny(num_classes_per_task=10, num_tasks=10, img_size=32, patch_size=4,
             quantize=True, **kw) -> QuantizedViT:
    """ViT-Tiny: 12 blocks, 3 heads, dim 192. ~5.7M params for CIFAR-32."""
    return QuantizedViT(img_size=img_size, patch_size=patch_size,
                        dim=192, depth=12, num_heads=3, mlp_ratio=4.0,
                        num_classes_per_task=num_classes_per_task,
                        num_tasks=num_tasks, quantize=quantize, **kw)


def vit_small(num_classes_per_task=10, num_tasks=10, img_size=32, patch_size=4,
              quantize=True, **kw) -> QuantizedViT:
    """ViT-Small: 12 blocks, 6 heads, dim 384. ~22M params for CIFAR-32."""
    return QuantizedViT(img_size=img_size, patch_size=patch_size,
                        dim=384, depth=12, num_heads=6, mlp_ratio=4.0,
                        num_classes_per_task=num_classes_per_task,
                        num_tasks=num_tasks, quantize=quantize, **kw)
