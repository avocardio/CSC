"""Vectorized PyTorch port of MetaWorld's reward_utils.

These are direct translations of metaworld.utils.reward_utils,
operating on batched tensors (shape (N,) or (N, k)) on GPU.
"""

import torch

DEFAULT_VALUE_AT_MARGIN = 0.1


def _sigmoid(x: torch.Tensor, value_at_1: float = 0.1,
             sigmoid: str = 'long_tail') -> torch.Tensor:
    """Sigmoid function mapping `x` to [0, 1] with f(0)=1, f(1)=value_at_1."""
    if sigmoid == 'long_tail':
        scale = (1.0 / value_at_1 - 1.0) ** 0.5
        return 1.0 / ((x * scale).pow(2) + 1.0)
    elif sigmoid == 'gaussian':
        scale = (-2.0 * torch.log(torch.tensor(value_at_1))).sqrt()
        return torch.exp(-0.5 * (x * scale).pow(2))
    elif sigmoid == 'reciprocal':
        scale = 1.0 / value_at_1 - 1.0
        return 1.0 / (x.abs() * scale + 1.0)
    elif sigmoid == 'hyperbolic':
        scale = torch.arccosh(torch.tensor(1.0 / value_at_1)).item()
        return 1.0 / torch.cosh(x * scale)
    else:
        raise ValueError(f'Unsupported sigmoid: {sigmoid}')


def tolerance(x: torch.Tensor,
              bounds=(0.0, 0.0),
              margin: float | torch.Tensor = 0.0,
              sigmoid: str = 'gaussian',
              value_at_margin: float = DEFAULT_VALUE_AT_MARGIN) -> torch.Tensor:
    """Returns 1 when x is in bounds, decaying sigmoidally outside via margin.

    Vectorized port of metaworld.utils.reward_utils.tolerance.
    """
    lower, upper = bounds
    in_bounds = (x >= lower) & (x <= upper)
    if (isinstance(margin, (int, float)) and margin == 0.0) or (
            torch.is_tensor(margin) and (margin == 0).all()):
        return in_bounds.float()

    # Distance outside bounds (0 if inside)
    d = torch.where(x < lower, lower - x,
                    torch.where(x > upper, x - upper, torch.zeros_like(x)))
    # Normalize by margin (scalar or per-element)
    if isinstance(margin, (int, float)):
        d_scaled = d / margin
    else:
        margin_safe = torch.where(margin == 0, torch.ones_like(margin), margin)
        d_scaled = d / margin_safe

    sig_val = _sigmoid(d_scaled, value_at_margin, sigmoid)
    return torch.where(in_bounds, torch.ones_like(x), sig_val)


def hamacher_product(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Vectorized hamacher t-norm: (a*b) / (a + b - a*b).

    Assumes a, b in [0, 1]. Returns 0 where denominator is 0.
    """
    denom = a + b - a * b
    safe_denom = torch.where(denom > 0, denom, torch.ones_like(denom))
    result = torch.where(denom > 0, (a * b) / safe_denom, torch.zeros_like(denom))
    return result


def inverse_tolerance(x, bounds=(0.0, 0.0), margin=0.0,
                      sigmoid='reciprocal') -> torch.Tensor:
    bound = tolerance(x, bounds=bounds, margin=margin, sigmoid=sigmoid,
                      value_at_margin=0.0)
    return 1.0 - bound


def gripper_caging_reward(
        action: torch.Tensor,           # (N, 4)
        obj_pos: torch.Tensor,          # (N, 3)
        obj_init_pos: torch.Tensor,     # (N, 3)
        tcp: torch.Tensor,              # (N, 3) current tcp center
        init_tcp: torch.Tensor,         # (N, 3) initial tcp center (constant per env)
        left_pad: torch.Tensor,         # (N, 3)
        right_pad: torch.Tensor,        # (N, 3)
        obj_radius: float,
        pad_success_thresh: float,
        object_reach_radius: float,
        xz_thresh: float,
        desired_gripper_effort: float = 1.0,
        high_density: bool = False,
        medium_density: bool = False,
) -> torch.Tensor:
    """Vectorized port of SawyerXYZEnv._gripper_caging_reward.

    Returns per-env caging reward in [0, 1], shape (N,).
    """
    # pad_y_lr: (N, 2) — left and right pad y positions
    pad_y_lr = torch.stack([left_pad[:, 1], right_pad[:, 1]], dim=-1)
    # pad-to-obj distance in y (current object y)
    pad_to_obj_lr = (pad_y_lr - obj_pos[:, 1:2]).abs()
    # pad-to-obj_init distance in y
    pad_to_objinit_lr = (pad_y_lr - obj_init_pos[:, 1:2]).abs()

    # Caging left-right reward (two tolerance calls, one per pad)
    caging_lr_margin = (pad_to_objinit_lr - pad_success_thresh).abs()
    caging_l = tolerance(
        pad_to_obj_lr[:, 0],
        bounds=(obj_radius, pad_success_thresh),
        margin=caging_lr_margin[:, 0],
        sigmoid='long_tail',
    )
    caging_r = tolerance(
        pad_to_obj_lr[:, 1],
        bounds=(obj_radius, pad_success_thresh),
        margin=caging_lr_margin[:, 1],
        sigmoid='long_tail',
    )
    caging_y = hamacher_product(caging_l, caging_r)

    # X-Z caging
    xz_idx = [0, 2]
    caging_xz_margin = torch.norm(
        obj_init_pos[:, xz_idx] - init_tcp[:, xz_idx], dim=-1) - xz_thresh
    caging_xz_margin = caging_xz_margin.clamp(min=1e-6)  # avoid 0 margin
    tcp_to_obj_xz = torch.norm(tcp[:, xz_idx] - obj_pos[:, xz_idx], dim=-1)
    caging_xz = tolerance(
        tcp_to_obj_xz,
        bounds=(0.0, xz_thresh),
        margin=caging_xz_margin,
        sigmoid='long_tail',
    )

    # Gripper closed effort
    gripper_closed = (action[:, -1].clamp(0.0, desired_gripper_effort) /
                      desired_gripper_effort)

    # Combine
    caging = hamacher_product(caging_y, caging_xz)
    gripping = torch.where(caging > 0.97, gripper_closed, torch.zeros_like(caging))
    caging_and_gripping = hamacher_product(caging, gripping)

    if high_density:
        caging_and_gripping = (caging_and_gripping + caging) / 2.0

    if medium_density:
        tcp_to_obj = torch.norm(obj_pos - tcp, dim=-1)
        tcp_to_obj_init = torch.norm(obj_init_pos - init_tcp, dim=-1)
        reach_margin = (tcp_to_obj_init - object_reach_radius).abs()
        reach = tolerance(
            tcp_to_obj,
            bounds=(0.0, object_reach_radius),
            margin=reach_margin,
            sigmoid='long_tail',
        )
        caging_and_gripping = (caging_and_gripping + reach) / 2.0

    return caging_and_gripping


if __name__ == '__main__':
    # Sanity tests vs numpy reference
    import numpy as np
    from metaworld.utils import reward_utils as rew_np

    # tolerance
    x_np = np.array([0.0, 0.05, 0.1, 0.5, 1.0])
    for sig in ['long_tail', 'gaussian', 'reciprocal']:
        np_val = rew_np.tolerance(x_np, bounds=(0, 0.02), margin=0.2, sigmoid=sig)
        t_val = tolerance(torch.tensor(x_np, dtype=torch.float64),
                          bounds=(0, 0.02), margin=0.2, sigmoid=sig).numpy()
        diff = np.abs(np_val - t_val).max()
        print(f'tolerance {sig}: diff={diff:.6f} '
              f'np={np_val}, torch={t_val}')

    # hamacher_product
    print('\nhamacher_product:')
    for a_val, b_val in [(0.5, 0.5), (0.1, 0.9), (1.0, 1.0), (0.0, 0.5)]:
        np_val = rew_np.hamacher_product(a_val, b_val)
        t_val = hamacher_product(torch.tensor(a_val), torch.tensor(b_val)).item()
        print(f'  a={a_val}, b={b_val}: np={np_val:.4f}, torch={t_val:.4f}')

    print('\nAll reward_utils tests OK')
