"""Channel and weight removal logic for self-compressing networks.

Handles:
- Detecting zero-bit channels
- Applying L1 penalty to biases of zero-bit channels
- Physically removing dead channels (and corresponding input channels in next layer)
- Updating optimizer state after removal
"""

import torch
import torch.nn as nn
from .quantization import DifferentiableQuantizer, get_quantizers


def apply_bias_l1_penalty(model, penalty_weight=0.01):
    """Apply L1 loss to biases of zero-bit channels.

    When a channel's bit depth reaches zero, its bias might still be nonzero.
    We apply L1 to push the bias toward zero before removing the channel.

    Returns:
        L1 loss term to add to the total loss.
    """
    l1_loss = torch.tensor(0.0, device=next(model.parameters()).device)
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer') and isinstance(module.quantizer, DifferentiableQuantizer):
            zero_mask = module.quantizer.get_zero_channels()
            if zero_mask.any() and hasattr(module, 'bias') and module.bias is not None:
                l1_loss = l1_loss + (module.bias[zero_mask].abs().sum())
    return penalty_weight * l1_loss


def find_removable_channels(model, bias_threshold=0.01):
    """Find channels that can be safely removed (b=0 AND bias~=0).

    Returns:
        Dict mapping module name -> list of channel indices to remove.
    """
    removable = {}
    for name, module in model.named_modules():
        if hasattr(module, 'quantizer') and isinstance(module.quantizer, DifferentiableQuantizer):
            zero_bit_mask = module.quantizer.get_zero_channels()
            if not zero_bit_mask.any():
                continue

            # Check if bias is also near zero
            if hasattr(module, 'bias') and module.bias is not None:
                bias_near_zero = module.bias.abs() < bias_threshold
                removable_mask = zero_bit_mask & bias_near_zero
            else:
                removable_mask = zero_bit_mask

            if removable_mask.any():
                indices = torch.where(removable_mask)[0].tolist()
                removable[name] = indices

    return removable


def remove_channels_from_conv(conv_module, channels_to_remove, remove_output=True):
    """Remove channels from a Conv2d layer.

    Args:
        conv_module: nn.Conv2d with .quantizer attribute
        channels_to_remove: list of channel indices to remove
        remove_output: if True, remove output channels; if False, remove input channels
    """
    if not channels_to_remove:
        return

    keep_mask = torch.ones(conv_module.out_channels if remove_output else conv_module.in_channels,
                           dtype=torch.bool)
    for idx in channels_to_remove:
        keep_mask[idx] = False

    with torch.no_grad():
        if remove_output:
            # Remove output channels
            new_weight = conv_module.weight.data[keep_mask]
            conv_module.weight = nn.Parameter(new_weight)
            conv_module.out_channels = new_weight.shape[0]

            if conv_module.bias is not None:
                conv_module.bias = nn.Parameter(conv_module.bias.data[keep_mask])

            # Update quantizer
            if hasattr(conv_module, 'quantizer'):
                q = conv_module.quantizer
                keep_indices = torch.where(keep_mask)[0]
                q.bit_depth = nn.Parameter(q.bit_depth.data[keep_indices])
                q.exponent = nn.Parameter(q.exponent.data[keep_indices])
                q.num_output_channels = new_weight.shape[0]
                q.weight_shape = new_weight.shape
                q.num_params = len(keep_indices)
                if len(new_weight.shape) == 4:
                    q.param_shape = (len(keep_indices), 1, 1, 1)
                else:
                    q.param_shape = (len(keep_indices),)
        else:
            # Remove input channels
            if conv_module.groups == 1:
                new_weight = conv_module.weight.data[:, keep_mask]
                conv_module.weight = nn.Parameter(new_weight)
                conv_module.in_channels = new_weight.shape[1]
                if hasattr(conv_module, 'quantizer'):
                    conv_module.quantizer.weight_shape = new_weight.shape


def remove_channels_from_batchnorm(bn_module, channels_to_remove):
    """Remove channels from a BatchNorm layer."""
    if not channels_to_remove:
        return

    keep_mask = torch.ones(bn_module.num_features, dtype=torch.bool)
    for idx in channels_to_remove:
        keep_mask[idx] = False

    with torch.no_grad():
        bn_module.weight = nn.Parameter(bn_module.weight.data[keep_mask])
        bn_module.bias = nn.Parameter(bn_module.bias.data[keep_mask])
        bn_module.running_mean = bn_module.running_mean[keep_mask]
        bn_module.running_var = bn_module.running_var[keep_mask]
        bn_module.num_features = keep_mask.sum().item()


def remove_dead_channels(model, optimizer=None, bias_threshold=0.01):
    """Remove all dead channels from the model.

    This is the main channel removal entry point. Finds removable channels,
    removes them from conv/bn layers, and updates optimizer state.

    Returns:
        Number of channels removed.
    """
    # For ResNet, channel removal is complex due to residual connections.
    # We only remove from layers where it's safe (no residual dimension mismatch).
    # The model should implement a .get_removable_layers() method that returns
    # pairs of (conv, next_layers) for safe removal.

    removable = find_removable_channels(model, bias_threshold)
    total_removed = 0

    if not removable:
        return 0

    # If model has custom removal logic, use it
    if hasattr(model, 'remove_channels'):
        total_removed = model.remove_channels(removable, optimizer)
    else:
        # Generic removal - just mask weights to zero for safety
        for name, module in model.named_modules():
            if name in removable and hasattr(module, 'quantizer'):
                zero_mask = module.quantizer.get_zero_channels()
                if zero_mask.any():
                    with torch.no_grad():
                        module.weight.data[zero_mask] = 0
                    total_removed += zero_mask.sum().item()

    # Rebuild optimizer state if channels were physically removed
    if total_removed > 0 and optimizer is not None:
        _rebuild_optimizer_state(optimizer, model)

    return total_removed


def _rebuild_optimizer_state(optimizer, model):
    """Rebuild optimizer after channel removal.

    The simplest approach: create new param groups with fresh state.
    This loses momentum but avoids stale state issues.
    """
    # Collect current param groups config
    param_groups_config = []
    for group in optimizer.param_groups:
        config = {k: v for k, v in group.items() if k != 'params'}
        param_groups_config.append(config)

    # Clear and recreate
    optimizer.state.clear()

    # Re-register parameters
    all_params = list(model.parameters())
    if len(param_groups_config) == 1:
        optimizer.param_groups[0]['params'] = all_params
    else:
        # Separate quantization params from weight params
        quant_params = []
        weight_params = []
        quant_param_ids = set()
        for m in model.modules():
            if isinstance(m, DifferentiableQuantizer):
                for p in m.parameters():
                    quant_param_ids.add(id(p))
                    quant_params.append(p)
        for p in model.parameters():
            if id(p) not in quant_param_ids:
                weight_params.append(p)

        if len(optimizer.param_groups) >= 2:
            optimizer.param_groups[0]['params'] = weight_params
            optimizer.param_groups[1]['params'] = quant_params
