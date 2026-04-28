"""Differentiable quantization module for self-compressing neural networks.

Implements the quantization function from Csefalvay & Imber (2023):
    q(x, b, e) = 2^e * round(min(max(2^(-e)*x, -2^(b-1)), 2^(b-1) - 1))

With STE for gradient propagation through rounding.
Supports channel-level, group-level, and weight-level granularity.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from enum import Enum


class CompressionGranularity(Enum):
    CHANNEL = "channel"
    GROUP = "group"
    WEIGHT = "weight"


class STERound(torch.autograd.Function):
    """Round with straight-through estimator for gradients.
    `setup_context` split out + generate_vmap_rule so the function works with
    functorch (vmap/grad), which is needed for the EWC Fisher pass when CSC is on."""
    generate_vmap_rule = True

    @staticmethod
    def forward(x):
        return torch.round(x)

    @staticmethod
    def setup_context(ctx, inputs, output):
        pass

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output


def ste_round(x):
    return STERound.apply(x)


def quantize(x, b, e):
    """Differentiable quantization function (eq. 1 from paper).

    Args:
        x: weights to quantize
        b: bit depth (>=0, real-valued during training)
        e: exponent (real-valued)

    Returns:
        Quantized weights. When b=0, output is zero.
    """
    b_clamped = b.clamp(min=0.0)
    # Clamp exponent to prevent 2**(-e) overflow when e drifts very negative
    # (which produces inf -> NaN through subsequent multiplications). The bound
    # ±20 covers any realistic dynamic range (2^20 ≈ 1e6) without saturating
    # inv_scale, while staying well below float32 overflow at 2^126.
    e_safe = e.clamp(min=-20.0, max=20.0)
    scale = (2.0 ** e_safe).clamp(min=1e-10)
    inv_scale = (2.0 ** (-e_safe))

    # Compute range bounds: [-2^(b-1), 2^(b-1) - 1]
    # For b=0: range is [-0.5, -0.5] which clamps everything to -0.5,
    # then rounds to 0 (or -1). We handle b=0 separately.
    half_range = 2.0 ** (b_clamped - 1)
    lower = -half_range
    upper = half_range - 1

    # Scale, clamp, round, unscale
    scaled = inv_scale * x
    clamped = torch.min(torch.max(scaled, lower), upper)
    rounded = ste_round(clamped)
    out = scale * rounded
    # Belt-and-braces: replace any non-finite slipping through (should be impossible
    # after the e_safe clamp above, but cheaply guards against future regressions)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


class DifferentiableQuantizer(nn.Module):
    """Applies differentiable quantization to a weight tensor.

    Manages learnable bit-depth (b) and exponent (e) parameters
    at the specified granularity level.
    """

    def __init__(self, weight_shape, granularity=CompressionGranularity.CHANNEL,
                 group_size=16, init_bit_depth=8.0, init_exponent=-4.0):
        """
        Args:
            weight_shape: shape of the weight tensor (O, I, H, W) for conv
            granularity: CHANNEL, GROUP, or WEIGHT level compression
            group_size: number of weights per group (only for GROUP granularity)
            init_bit_depth: initial bit depth for all parameters
            init_exponent: initial exponent for all parameters
        """
        super().__init__()
        self.weight_shape = weight_shape
        self.granularity = granularity
        self.group_size = group_size
        self.num_output_channels = weight_shape[0]

        if granularity == CompressionGranularity.CHANNEL:
            # One (b, e) pair per output channel
            num_params = weight_shape[0]
            if len(weight_shape) == 4:
                self.param_shape = (num_params, 1, 1, 1)
            elif len(weight_shape) == 2:
                self.param_shape = (num_params, 1)
            else:
                self.param_shape = (num_params,)
        elif granularity == CompressionGranularity.GROUP:
            # One (b, e) pair per group of weights within each channel
            weights_per_channel = 1
            for d in weight_shape[1:]:
                weights_per_channel *= d
            num_groups_per_channel = (weights_per_channel + group_size - 1) // group_size
            num_params = weight_shape[0] * num_groups_per_channel
            self.num_groups_per_channel = num_groups_per_channel
            self.weights_per_channel = weights_per_channel
            self.param_shape = (num_params,)
        elif granularity == CompressionGranularity.WEIGHT:
            # One (b, e) pair per weight
            num_params = 1
            for d in weight_shape:
                num_params *= d
            self.param_shape = weight_shape

        self.num_params = num_params
        self.init_bit_depth = float(init_bit_depth)
        self.bit_depth = nn.Parameter(torch.full((num_params,), init_bit_depth))
        self.exponent = nn.Parameter(torch.full((num_params,), init_exponent))

    def forward(self, weight):
        b = self.bit_depth.clamp(min=0.0)
        e = self.exponent

        if self.granularity == CompressionGranularity.CHANNEL:
            # Reshape b, e to broadcast over (I, H, W)
            b_shaped = b.view(self.param_shape)
            e_shaped = e.view(self.param_shape)
            return quantize(weight, b_shaped, e_shaped)

        elif self.granularity == CompressionGranularity.GROUP:
            O = self.weight_shape[0]
            flat = weight.view(O, -1)  # (O, I*H*W)
            # Pad to multiple of group_size
            W = flat.shape[1]
            pad = (self.group_size - W % self.group_size) % self.group_size
            if pad > 0:
                flat = F.pad(flat, (0, pad))
            # Reshape to (O, num_groups, group_size)
            flat = flat.view(O, self.num_groups_per_channel, self.group_size)
            b_shaped = b.view(O, self.num_groups_per_channel, 1)
            e_shaped = e.view(O, self.num_groups_per_channel, 1)
            quantized = quantize(flat, b_shaped, e_shaped)
            # Remove padding and reshape back
            quantized = quantized.view(O, -1)[:, :W]
            return quantized.view(self.weight_shape)

        elif self.granularity == CompressionGranularity.WEIGHT:
            b_shaped = b.view(self.param_shape)
            e_shaped = e.view(self.param_shape)
            return quantize(weight, b_shaped, e_shaped)

    def get_bit_depths(self):
        """Return clamped bit depths."""
        return self.bit_depth.clamp(min=0.0)

    def get_channel_bit_depths(self):
        """Return average bit depth per output channel."""
        b = self.get_bit_depths()
        if self.granularity == CompressionGranularity.CHANNEL:
            return b
        elif self.granularity == CompressionGranularity.GROUP:
            return b.view(self.num_output_channels, -1).mean(dim=1)
        elif self.granularity == CompressionGranularity.WEIGHT:
            return b.view(self.num_output_channels, -1).mean(dim=1)

    def get_zero_channels(self, threshold=0.01):
        """Return boolean mask of channels with near-zero bit depth."""
        channel_bits = self.get_channel_bit_depths()
        return channel_bits < threshold

    def compute_layer_bits(self):
        """Compute total bits for this layer (eq. 4 from paper)."""
        b = self.get_bit_depths()
        if self.granularity == CompressionGranularity.CHANNEL:
            # z_l = I * H * W * sum(b_i)
            weights_per_channel = 1
            for d in self.weight_shape[1:]:
                weights_per_channel *= d
            return weights_per_channel * b.sum()
        elif self.granularity == CompressionGranularity.GROUP:
            return b.sum() * self.group_size
        elif self.granularity == CompressionGranularity.WEIGHT:
            return b.sum()

    def compute_layer_bits_coupled(self, prev_bit_depths=None):
        """Compute layer bits with cross-layer coupling (eq. 5 from paper).

        Args:
            prev_bit_depths: bit depths of previous layer's output channels.
                If None, uses simple formula (eq. 4).
        """
        if prev_bit_depths is None:
            return self.compute_layer_bits()

        b = self.get_bit_depths()
        if self.granularity != CompressionGranularity.CHANNEL:
            return self.compute_layer_bits()

        if len(self.weight_shape) == 4:
            H, W = self.weight_shape[2], self.weight_shape[3]
        else:
            H, W = 1, 1

        # Indicator: which input channels (from prev layer) are alive
        prev_alive = (prev_bit_depths > 0.01).float()
        # Indicator: which output channels are alive
        curr_alive = (b > 0.01).float()

        # eq 5: H*W * (sum_j 1_{b'_j>0}) * (sum_i b_i) + H*W * (sum_i 1_{b_i>0}) * (sum_j b'_j)
        term1 = H * W * prev_alive.sum() * b.sum()
        term2 = H * W * curr_alive.sum() * prev_bit_depths.sum()
        return (term1 + term2) / 2  # Average of both perspectives


def compute_average_bit_depth(model, use_coupled=True):
    """Compute Q = (1/N) * sum(z_l) across all layers (eq. 3).

    Args:
        model: model with .quantizers attribute (list of DifferentiableQuantizer)
        use_coupled: whether to use cross-layer coupled formula (eq. 5)

    Returns:
        Q: average bit depth across entire network
    """
    total_bits = 0.0
    total_weights = 0

    quantizers = get_quantizers(model)
    prev_bits = None

    for quantizer in quantizers:
        n_weights = 1
        for d in quantizer.weight_shape:
            n_weights *= d
        total_weights += n_weights

        if use_coupled and prev_bits is not None:
            total_bits += quantizer.compute_layer_bits_coupled(prev_bits)
        else:
            total_bits += quantizer.compute_layer_bits()

        prev_bits = quantizer.get_channel_bit_depths()

    if total_weights == 0:
        return torch.tensor(0.0)

    return total_bits / total_weights


def get_quantizers(model):
    """Get all DifferentiableQuantizer modules from the model."""
    quantizers = []
    for module in model.modules():
        if isinstance(module, DifferentiableQuantizer):
            quantizers.append(module)
    return quantizers


def get_compression_stats(model):
    """Get compression statistics for the model."""
    quantizers = get_quantizers(model)
    total_weights = 0
    total_bits = 0.0
    zero_channels = 0
    total_channels = 0
    original_bits = 0

    for q in quantizers:
        n_weights = 1
        for d in q.weight_shape:
            n_weights *= d
        total_weights += n_weights
        total_bits += q.compute_layer_bits().item()
        original_bits += n_weights * 32  # 32-bit float baseline

        zero_mask = q.get_zero_channels()
        zero_channels += zero_mask.sum().item()
        total_channels += q.num_output_channels

    # Per-channel bit-depth histogram (for the "smart allocation" plot).
    bd_all = []
    init_bit = 8.0
    for q in quantizers:
        bd = q.get_channel_bit_depths().detach().cpu().clamp(min=0).tolist()
        bd_all.extend(bd)
        # capture init_bit from any quantizer (they all share it in our setup)
        init_bit = float(getattr(q, 'init_bit_depth', init_bit))
    return {
        'total_weights': total_weights,
        'total_bits': total_bits,
        'original_bits': original_bits,
        'compression_ratio': total_bits / original_bits if original_bits > 0 else 0,
        'avg_bit_depth': total_bits / total_weights if total_weights > 0 else 0,
        'init_bit_depth': init_bit,
        'utilization_8b': (total_bits / total_weights) / init_bit if total_weights > 0 else 0,
        'channel_bit_depths': bd_all,
        'zero_channels': zero_channels,
        'total_channels': total_channels,
        'channels_remaining_pct': 100 * (1 - zero_channels / total_channels) if total_channels > 0 else 100,
    }
