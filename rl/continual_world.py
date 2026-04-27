"""Continual World benchmark in PyTorch — clean reimplementation matching CW source.

Verified against the official TensorFlow reference at github.com/awarelab/continual_world
(Wolczyk et al. 2021). Architectural / protocol choices documented here:

- Multi-head actor and critic, shared quantized core (4x256 MLP).
  Actor heads: per-task (mu, log_std). Critic heads: per-task (Q1, Q2).
  Core: Dense(256) -> LayerNorm -> tanh, then 3x [Dense(256) -> LeakyReLU].
- Self-compression (CSC): real STE quantization applied to core weight matrices.
  Bit-depth and exponent are learnable per output channel; quant params have
  their own optimizer (lr=0.5, eps=1e-3) per Csefalvay & Imber (2023).
  Soft protection: gradients on core weights scaled by 1 / (1 + beta * acc_b)
  where acc_b is the per-channel max of bit-depth across completed tasks.
- log_alpha: one entry per task, never reset across tasks. target_entropy = -act_dim.
- log_std: clamp at [-20, 2] (CW reference uses clamp, not tanh+rescale).
- Buffer is reset on task change. Optimizers are rebuilt on task change.
- Done masking: only true terminations (term=True with ep_len < MAX_EP_LEN)
  store done=1; truncations store done=0.
- EWC: analytic Gaussian Fisher per sample via functorch
  F = (mu_g^2 + 2 * std_g^2) / std^2, summed over output dims, averaged over batch,
  clamped at 1e-5, accumulated additively across tasks. Lambda default 1e4.
- MAS: per-sample |grad| of (||mu||^2 + ||log_std||^2). Lambda default 1e4.
- L2: penalty on (theta - theta_prev_task)^2. Lambda default 1e5.
- PackNet: per-task ownership mask on core weights (heads excluded due to
  multi-head). Biases + LayerNorm frozen after task 0. Retrain step
  count = 100K (CW default). Retrain runs full SAC update.
- Eval: caches one env per task name to avoid re-creating MetaWorld envs.
"""

from __future__ import annotations
import os, sys, math, time, argparse, json
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metaworld
import gymnasium as gym  # noqa: F401  -- imported for side effects
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

from models.quantization import (
    DifferentiableQuantizer, CompressionGranularity, get_quantizers,
)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CW10_TASKS = [
    'hammer-v3', 'push-wall-v3', 'faucet-close-v3', 'push-back-v3',
    'stick-pull-v3', 'handle-press-side-v3', 'push-v3', 'shelf-place-v3',
    'window-close-v3', 'peg-unplug-side-v3',
]

OBS_DIM = 39
ACT_DIM = 4
MAX_EP_LEN = 200
LOG_STD_MIN, LOG_STD_MAX = -20.0, 2.0
HIDDEN = 256
N_CORE_LAYERS = 4


# ============================================================
# Quantized linear layer (channel-level CSC)
# ============================================================
class QuantizedLinear(nn.Module):
    """Linear layer with channel-wise differentiable quantization on weights."""

    def __init__(self, in_dim: int, out_dim: int, init_bit: float = 8.0,
                 init_exp: float = -4.0, quantize_enabled: bool = True):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        self.bias = nn.Parameter(torch.zeros(out_dim))
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        bound = 1.0 / math.sqrt(in_dim)
        nn.init.uniform_(self.bias, -bound, bound)
        self.quantizer = DifferentiableQuantizer(
            weight_shape=(out_dim, in_dim),
            granularity=CompressionGranularity.CHANNEL,
            init_bit_depth=init_bit, init_exponent=init_exp,
        )
        self.quantize_enabled = quantize_enabled

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        W = self.quantizer(self.weight) if self.quantize_enabled else self.weight
        return F.linear(x, W, self.bias)

    def channel_bit_depths(self) -> torch.Tensor:
        return self.quantizer.get_channel_bit_depths()


# ============================================================
# Replay buffer (per-transition task_id; reset on task change)
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int = 1_000_000, device: str = DEVICE):
        self.cap = capacity
        self.device = device
        self.obs = torch.zeros(capacity, OBS_DIM, device=device)
        self.act = torch.zeros(capacity, ACT_DIM, device=device)
        self.rew = torch.zeros(capacity, 1, device=device)
        self.nobs = torch.zeros(capacity, OBS_DIM, device=device)
        self.done = torch.zeros(capacity, 1, device=device)
        self.task = torch.zeros(capacity, dtype=torch.long, device=device)
        self.pos = 0
        self.size = 0

    def add(self, obs, act, rew, nobs, done, task_id):
        i = self.pos
        self.obs[i] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.act[i] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.rew[i] = float(rew)
        self.nobs[i] = torch.as_tensor(nobs, dtype=torch.float32, device=self.device)
        self.done[i] = float(done)
        self.task[i] = int(task_id)
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs: int):
        idx = torch.randint(0, self.size, (bs,), device=self.device)
        return (self.obs[idx], self.act[idx], self.rew[idx],
                self.nobs[idx], self.done[idx], self.task[idx])

    def reset(self):
        self.pos = 0
        self.size = 0

    def snapshot(self, n: int):
        n = min(n, self.size)
        idx = torch.randperm(self.size, device=self.device)[:n]
        return (self.obs[idx].clone(), self.act[idx].clone(),
                self.rew[idx].clone(), self.nobs[idx].clone(),
                self.done[idx].clone(), self.task[idx].clone())


class TaskReplayStore:
    """Per-task snapshot buffer used by replay/CSC variants."""

    def __init__(self, device: str = DEVICE):
        self.device = device
        self.data: list[tuple] = []

    def add_task(self, buf: ReplayBuffer, n: int = 10_000):
        self.data.append(buf.snapshot(n))

    def sample(self, bs: int):
        if not self.data:
            return None
        n_tasks = len(self.data)
        per = bs // n_tasks
        rem = bs - per * n_tasks
        parts = [[] for _ in range(6)]
        for k, d in enumerate(self.data):
            this_n = per + (1 if k < rem else 0)  # distribute remainder
            if this_n == 0:
                continue
            idx = torch.randint(0, d[0].shape[0], (this_n,), device=self.device)
            for j in range(6):
                parts[j].append(d[j][idx])
        return tuple(torch.cat(p) for p in parts)

    @property
    def n_tasks(self):
        return len(self.data)


# ============================================================
# Networks — multi-head actor / critic with shared (quantized) core
# ============================================================
class CWActor(nn.Module):
    def __init__(self, num_tasks: int, quantize: bool = True,
                 init_bit: float = 8.0, init_exp: float = -4.0):
        super().__init__()
        self.num_tasks = num_tasks
        self.fc1 = QuantizedLinear(OBS_DIM, HIDDEN, init_bit, init_exp, quantize)
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.fc2 = QuantizedLinear(HIDDEN, HIDDEN, init_bit, init_exp, quantize)
        self.fc3 = QuantizedLinear(HIDDEN, HIDDEN, init_bit, init_exp, quantize)
        self.fc4 = QuantizedLinear(HIDDEN, HIDDEN, init_bit, init_exp, quantize)
        # Per-task heads: weight (T, A, H), bias (T, A)
        self.heads_mu_W = nn.Parameter(torch.empty(num_tasks, ACT_DIM, HIDDEN))
        self.heads_mu_b = nn.Parameter(torch.zeros(num_tasks, ACT_DIM))
        self.heads_ls_W = nn.Parameter(torch.empty(num_tasks, ACT_DIM, HIDDEN))
        self.heads_ls_b = nn.Parameter(torch.zeros(num_tasks, ACT_DIM))
        for W in (self.heads_mu_W, self.heads_ls_W):
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    def core_layers(self):
        return [self.fc1, self.fc2, self.fc3, self.fc4]

    def core(self, obs: torch.Tensor) -> torch.Tensor:
        h = torch.tanh(self.ln1(self.fc1(obs)))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        return h

    def forward(self, obs: torch.Tensor, task_id):
        h = self.core(obs)
        mu, log_std = self._heads(h, task_id)
        return mu, log_std

    def _heads(self, h: torch.Tensor, task_id):
        if isinstance(task_id, int):
            mu_W = self.heads_mu_W[task_id]              # (A, H)
            mu_b = self.heads_mu_b[task_id]              # (A,)
            ls_W = self.heads_ls_W[task_id]
            ls_b = self.heads_ls_b[task_id]
            mu = h @ mu_W.t() + mu_b
            log_std = h @ ls_W.t() + ls_b
        else:
            # task_id: (B,) tensor
            mu_W = self.heads_mu_W[task_id]              # (B, A, H)
            mu_b = self.heads_mu_b[task_id]              # (B, A)
            ls_W = self.heads_ls_W[task_id]
            ls_b = self.heads_ls_b[task_id]
            mu = torch.einsum('bah,bh->ba', mu_W, h) + mu_b
            log_std = torch.einsum('bah,bh->ba', ls_W, h) + ls_b
        log_std = log_std.clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs: torch.Tensor, task_id):
        mu, log_std = self.forward(obs, task_id)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        # Numerically stable squash correction (Haarnoja 2018)
        lp = dist.log_prob(x) - (2 * (math.log(2.0) - x - F.softplus(-2 * x)))
        return action, lp.sum(-1, keepdim=True), mu

    @torch.no_grad()
    def act_stochastic(self, obs_np: np.ndarray, task_id: int) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32,
                              device=self.heads_mu_W.device).unsqueeze(0)
        a, _, _ = self.sample(obs, task_id)
        return a.cpu().numpy().flatten()

    @torch.no_grad()
    def act_deterministic(self, obs_np: np.ndarray, task_id: int) -> np.ndarray:
        obs = torch.as_tensor(obs_np, dtype=torch.float32,
                              device=self.heads_mu_W.device).unsqueeze(0)
        mu, _ = self.forward(obs, task_id)
        return torch.tanh(mu).cpu().numpy().flatten()

    def core_param_groups(self):
        """Yield (weight_params, quant_params) of the core."""
        weight_params, quant_params = [], []
        for layer in self.core_layers():
            weight_params.append(layer.weight)
            weight_params.append(layer.bias)
            for p in layer.quantizer.parameters():
                quant_params.append(p)
        # LayerNorm and heads: pure weight params
        weight_params.extend(list(self.ln1.parameters()))
        weight_params.append(self.heads_mu_W); weight_params.append(self.heads_mu_b)
        weight_params.append(self.heads_ls_W); weight_params.append(self.heads_ls_b)
        return weight_params, quant_params

    def compression_loss(self) -> torch.Tensor:
        """Q = (1/N) * sum_l (in_dim_l * sum_i b_{i,l}). Avg bit-depth per weight."""
        total_bits = 0.0
        total_weights = 0
        for layer in self.core_layers():
            n_weights = layer.in_dim * layer.out_dim
            b_sum = layer.channel_bit_depths().sum()
            total_bits = total_bits + layer.in_dim * b_sum
            total_weights += n_weights
        return total_bits / max(total_weights, 1)


class CWCritic(nn.Module):
    """Per-task Q1/Q2 heads with shared full-precision core."""

    def __init__(self, num_tasks: int):
        super().__init__()
        d = OBS_DIM + ACT_DIM
        self.fc1 = nn.Linear(d, HIDDEN)
        self.ln1 = nn.LayerNorm(HIDDEN)
        self.fc2 = nn.Linear(HIDDEN, HIDDEN)
        self.fc3 = nn.Linear(HIDDEN, HIDDEN)
        self.fc4 = nn.Linear(HIDDEN, HIDDEN)
        # Per-task heads (T, 1, H)
        self.q1_W = nn.Parameter(torch.empty(num_tasks, 1, HIDDEN))
        self.q1_b = nn.Parameter(torch.zeros(num_tasks, 1))
        self.q2_W = nn.Parameter(torch.empty(num_tasks, 1, HIDDEN))
        self.q2_b = nn.Parameter(torch.zeros(num_tasks, 1))
        for W in (self.q1_W, self.q2_W):
            nn.init.kaiming_uniform_(W, a=math.sqrt(5))

    def core(self, obs, act):
        x = torch.cat([obs, act], -1)
        h = torch.tanh(self.ln1(self.fc1(x)))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        return h

    def forward(self, obs, act, task_id):
        h = self.core(obs, act)
        if isinstance(task_id, int):
            q1 = h @ self.q1_W[task_id].t() + self.q1_b[task_id]
            q2 = h @ self.q2_W[task_id].t() + self.q2_b[task_id]
        else:
            q1 = torch.einsum('boh,bh->bo', self.q1_W[task_id], h) + self.q1_b[task_id]
            q2 = torch.einsum('boh,bh->bo', self.q2_W[task_id], h) + self.q2_b[task_id]
        return q1, q2


# ============================================================
# SAC + CL agent
# ============================================================
class SACAgent:
    def __init__(self, method: str = 'finetune', n_tasks: int = 10,
                 lr: float = 1e-3, lr_quant: float = 0.5,
                 gamma: float = 0.99, polyak: float = 0.995,
                 batch_size: int = 128, cl_reg_coef: float = 1e4,
                 replay_ratio: float = 0.5, replay_per_task: int = 10_000,
                 gamma_comp: float = 1e-3, grad_scale_beta: float = 1.0,
                 bit_floor: float = 1.0, reset_alpha: bool = False,
                 device: str = DEVICE):
        self.device = device
        self.method = method
        self.n_tasks = n_tasks
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.cl_reg_coef = cl_reg_coef
        self.replay_ratio = replay_ratio
        self.replay_per_task = replay_per_task
        self.gamma_comp = gamma_comp
        self.grad_scale_beta = grad_scale_beta
        self.bit_floor = bit_floor
        self.reset_alpha = reset_alpha
        self.lr = lr
        self.lr_quant = lr_quant

        self.use_csc = method == 'csc'
        self.use_replay = method in ('replay', 'csc', 'ewc_replay')
        self.use_ewc = method in ('ewc', 'ewc_replay')
        self.use_mas = method == 'mas'
        self.use_l2 = method == 'l2'
        self.use_packnet = method == 'packnet'

        self.actor = CWActor(n_tasks, quantize=self.use_csc).to(device)
        self.critic = CWCritic(n_tasks).to(device)        # contains twin Q1/Q2
        self.critic_t = CWCritic(n_tasks).to(device)
        self.critic_t.load_state_dict(self.critic.state_dict())
        for p in self.critic_t.parameters(): p.requires_grad = False

        # Per-task log_alpha, never reset (CW default)
        self.target_entropy = -float(ACT_DIM)
        self.log_alpha = nn.Parameter(torch.zeros(n_tasks, device=device))

        self._build_optimizers()

        self.replay_store = TaskReplayStore(device) if self.use_replay else None
        self._current_task = 0

        # CL state
        self.ewc_fisher: dict[str, torch.Tensor] = {}
        self.ewc_params: dict[str, torch.Tensor] = {}
        self.mas_imp: dict[str, torch.Tensor] = {}
        self.mas_params: dict[str, torch.Tensor] = {}
        self.l2_params: dict[str, torch.Tensor] = {}
        # CSC accumulated bit-depth per core layer (per output channel)
        self.acc_bits = ([torch.zeros(HIDDEN, device=device)
                          for _ in range(N_CORE_LAYERS)] if self.use_csc else [])
        # PackNet ownership: weight matrix only (heads excluded since multi-head)
        self.pn_owner: dict[str, torch.Tensor] = {}
        self.pn_freeze_bias = False
        if self.use_packnet:
            for layer in self.actor.core_layers():
                self.pn_owner[id(layer.weight)] = torch.zeros_like(
                    layer.weight, dtype=torch.int8)
        self.pn_retrain_steps = 100_000

    # ---- optimizer construction ----
    def _build_optimizers(self):
        weight_params, quant_params = self.actor.core_param_groups()
        # Use AdamW with eps split for quantization stability (per supervised paper)
        self.actor_opt = torch.optim.Adam(weight_params, lr=self.lr, eps=1e-5)
        if self.use_csc:
            self.imp_opt = torch.optim.Adam(quant_params, lr=self.lr_quant, eps=1e-3)
        else:
            self.imp_opt = None
        self.critic_opt = torch.optim.Adam(
            self.critic.parameters(), lr=self.lr, eps=1e-5)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def reset_for_new_task(self, task_idx: int):
        """Rebuild optimizers (CW default). Optionally reset alpha for that task."""
        self._build_optimizers()
        if self.reset_alpha:
            with torch.no_grad():
                self.log_alpha.data[task_idx] = 0.0
        self._current_task = task_idx

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.detach().exp()

    # ---- update ----
    def update(self, buffer: ReplayBuffer):
        # Sample batch (current task only or mixed)
        if self.replay_store and self.replay_store.n_tasks > 0:
            n_curr = int(self.batch_size * (1 - self.replay_ratio))
            n_rep = self.batch_size - n_curr
            sc, ac, rc, nsc, dc, tc = buffer.sample(n_curr)
            rep = self.replay_store.sample(n_rep)
            s = torch.cat([sc, rep[0]]); a = torch.cat([ac, rep[1]])
            r = torch.cat([rc, rep[2]]); ns = torch.cat([nsc, rep[3]])
            d = torch.cat([dc, rep[4]]); t = torch.cat([tc, rep[5]])
        else:
            s, a, r, ns, d, t = buffer.sample(self.batch_size)

        alpha_per_sample = self.log_alpha.detach().exp()[t].unsqueeze(-1)  # (B, 1)

        # ---- critic update ----
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(ns, t)
            q1t, q2t = self.critic_t(ns, na, t)
            qt = r + (1 - d) * self.gamma * (
                torch.min(q1t, q2t) - alpha_per_sample * nlp)
        q1p, q2p = self.critic(s, a, t)
        critic_loss = 0.5 * F.mse_loss(q1p, qt) + 0.5 * F.mse_loss(q2p, qt)
        self.critic_opt.zero_grad(set_to_none=True)
        critic_loss.backward()
        self.critic_opt.step()

        # ---- actor update ----
        na2, lp2, _ = self.actor.sample(s, t)
        q1pi, q2pi = self.critic(s, na2, t)
        actor_loss = (alpha_per_sample * lp2 - torch.min(q1pi, q2pi)).mean()

        if self.use_csc:
            actor_loss = actor_loss + self.gamma_comp * self.actor.compression_loss()
        if self.use_ewc and self.ewc_fisher:
            ewc_pen = sum(
                (self.ewc_fisher[n] * (p - self.ewc_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.ewc_fisher
            )
            actor_loss = actor_loss + self.cl_reg_coef * ewc_pen
        if self.use_mas and self.mas_imp:
            mas_pen = sum(
                (self.mas_imp[n] * (p - self.mas_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.mas_imp
            )
            actor_loss = actor_loss + self.cl_reg_coef * mas_pen
        if self.use_l2 and self.l2_params:
            l2_pen = sum(
                (p - self.l2_params[n]).pow(2).sum()
                for n, p in self.actor.named_parameters() if n in self.l2_params
            )
            actor_loss = actor_loss + self.cl_reg_coef * l2_pen

        self.actor_opt.zero_grad(set_to_none=True)
        if self.imp_opt is not None:
            self.imp_opt.zero_grad(set_to_none=True)
        actor_loss.backward()

        if self.use_csc and self.grad_scale_beta > 0:
            self._scale_core_grads_by_acc_bits()
        if self.use_packnet:
            self._packnet_mask_grads()

        self.actor_opt.step()
        if self.imp_opt is not None:
            self.imp_opt.step()
            # Floor on bit-depth: prevents permanent dead channels (b=0 has zero
            # gradient so once crossed, channel never recovers). Floor at 1 keeps
            # heavy quantization possible but channel still has gradient flow.
            if self.bit_floor > 0:
                with torch.no_grad():
                    for layer in self.actor.core_layers():
                        layer.quantizer.bit_depth.data.clamp_(min=self.bit_floor)
                        # Bound exponent so 2**(-e) cannot overflow to inf and
                        # poison the actor with NaN. Without this, e can drift
                        # past -126 over many updates with lr_quant=0.5 and
                        # produce unreproducible seed-dependent NaN crashes.
                        layer.quantizer.exponent.data.clamp_(min=-20.0, max=20.0)

        # ---- alpha update (CURRENT-task slice only) ----
        # Note: with multi-head + per-task log_alpha, computing the alpha update
        # over the mixed (current + replay) batch lets old-task alphas keep
        # receiving gradient. For an old task whose head has converged near the
        # log_std clamp (so the policy is near-deterministic), lp(sample) becomes
        # large positive and the gradient pushes log_alpha[old] up unboundedly,
        # producing alpha=1e6 nonsense. The fix: only train alpha on the
        # current-task slice of the batch. This matches the spirit of CW's
        # reservoir replay where the data is dominated by the current task.
        cur_mask = (t == self._current_task)
        if cur_mask.any():
            cur_lp = lp2.squeeze(-1)[cur_mask]
            cur_alpha = self.log_alpha[self._current_task].exp()
            alpha_loss = -(cur_alpha *
                           (cur_lp + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad(set_to_none=True)
            alpha_loss.backward()
            self.alpha_opt.step()

        # ---- target update ----
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_t.parameters()):
                tp.data.mul_(self.polyak).add_(p.data, alpha=1 - self.polyak)

    # ---- helpers ----
    def _scale_core_grads_by_acc_bits(self):
        """Soft protection: scale core weight gradients by 1 / (1 + beta * acc_b)."""
        for i, layer in enumerate(self.actor.core_layers()):
            acc = self.acc_bits[i]                                # (out_dim,)
            scale = 1.0 / (1.0 + self.grad_scale_beta * acc)
            if layer.weight.grad is not None:
                layer.weight.grad.mul_(scale.unsqueeze(1))
            if layer.bias.grad is not None:
                layer.bias.grad.mul_(scale)

    def _packnet_mask_grads(self):
        for layer in self.actor.core_layers():
            owner = self.pn_owner[id(layer.weight)]
            if layer.weight.grad is not None:
                mask = (owner == self._current_task).float()
                layer.weight.grad.mul_(mask)
        if self.pn_freeze_bias:
            # Freeze biases (core + ln) and LN params after task 0
            for layer in self.actor.core_layers():
                if layer.bias.grad is not None:
                    layer.bias.grad.zero_()
            for p in self.actor.ln1.parameters():
                if p.grad is not None:
                    p.grad.zero_()

    # ---- task boundary work ----
    def on_task_end(self, buffer: ReplayBuffer, task_idx: int):
        if self.use_ewc:
            self._compute_ewc_fisher(buffer, task_idx)
        if self.use_mas:
            self._compute_mas_importance(buffer, task_idx)
        if self.use_l2:
            for n, p in self.actor.named_parameters():
                self.l2_params[n] = p.data.clone()
        if self.replay_store is not None:
            self.replay_store.add_task(buffer, self.replay_per_task)
        if self.use_csc:
            with torch.no_grad():
                for i, layer in enumerate(self.actor.core_layers()):
                    b = layer.channel_bit_depths().clamp(min=0)
                    self.acc_bits[i] = torch.max(self.acc_bits[i], b)
        if self.use_packnet:
            self._packnet_prune(task_idx)
            self._packnet_retrain(buffer)
            if task_idx == 0:
                self.pn_freeze_bias = True

    def _compute_ewc_fisher(self, buffer: ReplayBuffer, task_idx: int,
                            n_batches: int = 10, bs: int = 256):
        """Analytic Gaussian Fisher per-sample (CW reference formula).

        F = sum_j [(d_mu_j/d_theta)^2 + 2*(d_std_j/d_theta)^2] / std_j^2
        averaged over batch, clamped at 1e-5, accumulated additively.
        """
        from torch.func import functional_call, vmap, grad

        params = {n: p.detach() for n, p in self.actor.named_parameters()}
        names = [n for n in params if 'quantizer' not in n]
        for n in names:
            self.ewc_params[n] = params[n].clone()

        fisher_acc = {n: torch.zeros_like(params[n]) for n in names}

        def f_mu_j(p_dict, sample, j, t_id):
            mu, _ = functional_call(self.actor, p_dict,
                                    (sample.unsqueeze(0), int(t_id)))
            return mu.squeeze(0)[j]

        def f_ls_j(p_dict, sample, j, t_id):
            _, log_std = functional_call(self.actor, p_dict,
                                         (sample.unsqueeze(0), int(t_id)))
            return log_std.squeeze(0)[j]

        # Sample n_batches mini-batches from current task data
        t_id = task_idx
        for _ in range(n_batches):
            s, _, _, _, _, _ = buffer.sample(bs)
            with torch.no_grad():
                _, log_std_b = self.actor(s, t_id)
                std_b = log_std_b.exp().clamp(min=1e-3)            # (bs, A)

            for j in range(ACT_DIM):
                g_mu = vmap(grad(f_mu_j), in_dims=(None, 0, None, None))(
                    params, s, j, t_id)
                g_ls = vmap(grad(f_ls_j), in_dims=(None, 0, None, None))(
                    params, s, j, t_id)
                std_j = std_b[:, j]                                 # (bs,)
                for n in names:
                    if n not in g_mu:
                        continue
                    mg = g_mu[n]                                    # (bs, *p_shape)
                    lg = g_ls[n]
                    extra = mg.dim() - 1
                    sj = std_j.view(-1, *([1] * extra))             # broadcast
                    sg = sj * lg                                    # std grad = std * d(log_std)
                    f = (mg.pow(2) + 2 * sg.pow(2)) / (sj.pow(2) + 1e-6)
                    fisher_acc[n] = fisher_acc[n] + f.sum(0)

        denom = n_batches * bs
        for n in names:
            f = (fisher_acc[n] / denom).clamp(min=1e-5)
            self.ewc_fisher[n] = self.ewc_fisher.get(n, torch.zeros_like(f)) + f
        # Diagnostic
        max_f = max(f.max().item() for f in self.ewc_fisher.values())
        mean_f = sum(f.mean().item() for f in self.ewc_fisher.values()) / len(self.ewc_fisher)
        print(f'    EWC Fisher: max={max_f:.4f} mean={mean_f:.4f}', flush=True)

    def _compute_mas_importance(self, buffer: ReplayBuffer, task_idx: int,
                                n_batches: int = 10, bs: int = 256):
        """Per-sample |grad| of (||mu||^2 + ||log_std||^2)."""
        from torch.func import functional_call, vmap, grad

        params = {n: p.detach() for n, p in self.actor.named_parameters()}
        names = [n for n in params if 'quantizer' not in n]
        for n in names:
            self.mas_params[n] = params[n].clone()

        imp_acc = {n: torch.zeros_like(params[n]) for n in names}

        def f_norm(p_dict, sample, t_id):
            mu, log_std = functional_call(self.actor, p_dict,
                                          (sample.unsqueeze(0), int(t_id)))
            return mu.pow(2).sum() + log_std.pow(2).sum()

        for _ in range(n_batches):
            s, _, _, _, _, _ = buffer.sample(bs)
            g = vmap(grad(f_norm), in_dims=(None, 0, None))(params, s, task_idx)
            for n in names:
                if n in g:
                    imp_acc[n] = imp_acc[n] + g[n].abs().sum(0)
        denom = n_batches * bs
        for n in names:
            imp = (imp_acc[n] / denom).clamp(min=1e-5)
            self.mas_imp[n] = self.mas_imp.get(n, torch.zeros_like(imp)) + imp

    def _packnet_prune(self, task_idx: int):
        if task_idx >= self.n_tasks - 1:
            return
        tasks_left = self.n_tasks - task_idx - 1
        prune_perc = tasks_left / (tasks_left + 1)
        with torch.no_grad():
            for layer in self.actor.core_layers():
                W = layer.weight
                owner = self.pn_owner[id(W)]
                mask = (owner == task_idx)
                vals = W[mask].abs()
                if vals.numel() == 0:
                    continue
                k = int(vals.numel() * prune_perc)
                if k == 0:
                    continue
                threshold = vals.sort().values[k]
                prune_mask = mask & (W.abs() <= threshold)
                W[prune_mask] = 0.0
                owner[prune_mask] = task_idx + 1
        print(f'    PackNet: pruned {prune_perc:.1%} of task-{task_idx} weights',
              flush=True)

    def _packnet_retrain(self, buffer: ReplayBuffer):
        # Fresh optimizer for retrain (CW protocol)
        self._build_optimizers()
        for _ in range(self.pn_retrain_steps):
            self.update(buffer)
        self._build_optimizers()
        print(f'    PackNet: retrained for {self.pn_retrain_steps} steps', flush=True)


# ============================================================
# Environment + evaluation
# ============================================================
_ENV_CACHE: dict[str, object] = {}


def make_env(task_name: str):
    """Cache one env per task to avoid MetaWorld ML1 setup overhead."""
    env = _ENV_CACHE.get(task_name)
    if env is None:
        ml1 = metaworld.ML1(task_name)
        env = ml1.train_classes[task_name]()
        env.set_task(ml1.train_tasks[0])
        _ENV_CACHE[task_name] = env
    return env


def evaluate(actor: CWActor, task_name: str, task_idx: int,
             n_episodes: int = 10, deterministic: bool = False) -> float:
    env = make_env(task_name)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_success = False
        for step in range(MAX_EP_LEN):
            if deterministic:
                a = actor.act_deterministic(obs, task_idx)
            else:
                a = actor.act_stochastic(obs, task_idx)
            obs, _, term, trunc, info = env.step(a)
            if info.get('success', 0):
                ep_success = True
            if term or step >= MAX_EP_LEN - 1:
                break
        if ep_success:
            successes += 1
    return successes / n_episodes


# ============================================================
# Checkpointing
# ============================================================
def save_checkpoint(path: str, agent: SACAgent, task_idx: int,
                    eval_history: list, results: dict, rng_state: dict):
    state = {
        'task_idx': task_idx,
        'method': agent.method,
        'actor': agent.actor.state_dict(),
        'critic': agent.critic.state_dict(),
        'critic_t': agent.critic_t.state_dict(),
        'log_alpha': agent.log_alpha.data.clone(),
        'actor_opt': agent.actor_opt.state_dict(),
        'critic_opt': agent.critic_opt.state_dict(),
        'alpha_opt': agent.alpha_opt.state_dict(),
        'imp_opt': (agent.imp_opt.state_dict() if agent.imp_opt is not None else None),
        'replay_data': ([tuple(t.cpu() for t in d) for d in agent.replay_store.data]
                        if agent.replay_store else None),
        'ewc_fisher': {k: v.cpu() for k, v in agent.ewc_fisher.items()},
        'ewc_params': {k: v.cpu() for k, v in agent.ewc_params.items()},
        'mas_imp': {k: v.cpu() for k, v in agent.mas_imp.items()},
        'mas_params': {k: v.cpu() for k, v in agent.mas_params.items()},
        'l2_params': {k: v.cpu() for k, v in agent.l2_params.items()},
        'acc_bits': [a.cpu() for a in agent.acc_bits],
        'pn_owner': {k: v.cpu() for k, v in agent.pn_owner.items()},
        'pn_freeze_bias': agent.pn_freeze_bias,
        'eval_history': eval_history,
        'results': results,
        'rng_state': rng_state,
    }
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, agent: SACAgent):
    s = torch.load(path, map_location=agent.device, weights_only=False)
    agent.actor.load_state_dict(s['actor'])
    agent.critic.load_state_dict(s['critic'])
    agent.critic_t.load_state_dict(s['critic_t'])
    agent.log_alpha.data.copy_(s['log_alpha'])
    agent.actor_opt.load_state_dict(s['actor_opt'])
    agent.critic_opt.load_state_dict(s['critic_opt'])
    agent.alpha_opt.load_state_dict(s['alpha_opt'])
    if s['imp_opt'] is not None and agent.imp_opt is not None:
        agent.imp_opt.load_state_dict(s['imp_opt'])
    if s['replay_data'] is not None and agent.replay_store is not None:
        agent.replay_store.data = [tuple(t.to(agent.device) for t in d)
                                   for d in s['replay_data']]
    agent.ewc_fisher = {k: v.to(agent.device) for k, v in s['ewc_fisher'].items()}
    agent.ewc_params = {k: v.to(agent.device) for k, v in s['ewc_params'].items()}
    agent.mas_imp = {k: v.to(agent.device) for k, v in s['mas_imp'].items()}
    agent.mas_params = {k: v.to(agent.device) for k, v in s['mas_params'].items()}
    agent.l2_params = {k: v.to(agent.device) for k, v in s['l2_params'].items()}
    agent.acc_bits = [a.to(agent.device) for a in s['acc_bits']]
    agent.pn_owner = {k: v.to(agent.device) for k, v in s['pn_owner'].items()}
    agent.pn_freeze_bias = s['pn_freeze_bias']
    return s['task_idx'], s['eval_history'], s['results'], s['rng_state']


# ============================================================
# Training
# ============================================================
UPDATE_EVERY = 50
UPDATE_AFTER = 1000
START_STEPS = 10_000


def train(method: str, tasks: list[str], steps_per_task: int, seed: int,
          cl_reg_coef: float, replay_ratio: float, replay_per_task: int,
          gamma_comp: float, grad_scale_beta: float, lr_quant: float,
          bit_floor: float, reset_alpha: bool,
          ckpt_path: str | None, eval_every: int = 100_000):
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SACAgent(method=method, n_tasks=len(tasks),
                     cl_reg_coef=cl_reg_coef,
                     replay_ratio=replay_ratio,
                     replay_per_task=replay_per_task,
                     gamma_comp=gamma_comp,
                     grad_scale_beta=grad_scale_beta,
                     lr_quant=lr_quant,
                     bit_floor=bit_floor,
                     reset_alpha=reset_alpha)
    buffer = ReplayBuffer()

    results: dict = {}
    eval_history: list = []
    start_task = 0

    if ckpt_path and os.path.isfile(ckpt_path):
        start_task, eval_history, results, rng = load_checkpoint(ckpt_path, agent)
        torch.set_rng_state(rng['torch'])
        np.random.set_state(rng['numpy'])
        start_task += 1                                # resume on next task
        print(f'Resumed from {ckpt_path}: starting at task {start_task}', flush=True)

    t_global = time.time()

    for task_idx in range(start_task, len(tasks)):
        task_name = tasks[task_idx]
        print(f'\n{"="*60}\nTASK {task_idx}: {task_name} ({steps_per_task:,} steps)\n{"="*60}',
              flush=True)
        env = make_env(task_name)
        obs, _ = env.reset()
        ep_len = 0
        buffer.reset()
        agent.reset_for_new_task(task_idx)
        t0 = time.time()
        ret_sum = 0.0
        ep_returns: list[float] = []

        for step in range(steps_per_task):
            if step < START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.actor.act_stochastic(obs, task_idx)
            next_obs, reward, term, trunc, info = env.step(action)
            ep_len += 1
            ret_sum += float(reward)
            done = 1.0 if (term and ep_len < MAX_EP_LEN) else 0.0
            buffer.add(obs, action, reward, next_obs, done, task_idx)
            obs = next_obs
            if term or ep_len >= MAX_EP_LEN:
                ep_returns.append(ret_sum)
                ret_sum = 0.0
                ep_len = 0
                obs, _ = env.reset()

            if step >= UPDATE_AFTER and step % UPDATE_EVERY == 0:
                for _ in range(UPDATE_EVERY):                # UTD = 1.0
                    agent.update(buffer)

            if (step + 1) % eval_every == 0:
                evals = {tasks[ti]: evaluate(agent.actor, tasks[ti], ti)
                         for ti in range(task_idx + 1)}
                eval_history.append({'task_idx': task_idx, 'step': step + 1,
                                     'evals': evals})
                elapsed = time.time() - t0
                sps = (step + 1) / max(elapsed, 1)
                ret_avg = (sum(ep_returns[-10:]) / max(len(ep_returns[-10:]), 1)
                           if ep_returns else 0)
                parts = ' '.join(f'{tasks[ti][:6]}={evals[tasks[ti]]:.2f}'
                                 for ti in range(task_idx + 1))
                rep = f' rep={agent.replay_store.n_tasks}t' if agent.replay_store else ''
                a_eff = agent.alpha[task_idx].item()
                print(f'  {(step+1)//1000}K ({sps:.0f} sps) {parts}'
                      f'{rep} alpha={a_eff:.3f} ret={ret_avg:.1f}',
                      flush=True)

        agent.on_task_end(buffer, task_idx)

        if ckpt_path:
            rng_state = {'torch': torch.get_rng_state(),
                         'numpy': np.random.get_state()}
            save_checkpoint(ckpt_path, agent, task_idx, eval_history, results,
                            rng_state)
            print(f'  Checkpoint -> {ckpt_path}', flush=True)

    # Final eval (more episodes)
    print(f'\n{"="*60}\nFINAL ({time.time()-t_global:.0f}s)\n{"="*60}', flush=True)
    final = {tasks[ti]: evaluate(agent.actor, tasks[ti], ti, n_episodes=20)
             for ti in range(len(tasks))}
    for tn in tasks:
        print(f'  {tn}: {final[tn]:.2f}', flush=True)
    avg = float(np.mean(list(final.values())))
    print(f'  Average: {avg:.2f}', flush=True)
    results['final'] = final
    results['eval_history'] = eval_history
    results['avg'] = avg
    return results


# ============================================================
# CLI
# ============================================================
def main():
    p = argparse.ArgumentParser()
    p.add_argument('--method', required=True,
                   choices=['finetune', 'l2', 'ewc', 'mas', 'replay',
                            'ewc_replay', 'packnet', 'csc'])
    p.add_argument('--steps_per_task', type=int, default=1_000_000)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--cl_reg_coef', type=float, default=1e4,
                   help='1e4 for EWC/MAS, 1e5 for L2 (CW defaults)')
    p.add_argument('--replay_ratio', type=float, default=0.5)
    p.add_argument('--replay_per_task', type=int, default=10_000)
    p.add_argument('--gamma_comp', type=float, default=1e-3)
    p.add_argument('--grad_scale_beta', type=float, default=1.0)
    p.add_argument('--lr_quant', type=float, default=0.5)
    p.add_argument('--bit_floor', type=float, default=1.0)
    p.add_argument('--reset_alpha', action='store_true')
    p.add_argument('--tasks', default='cw10', choices=['cw10', 'cw20', 'cw_subset'])
    p.add_argument('--eval_every', type=int, default=100_000)
    p.add_argument('--ckpt', default='', help='checkpoint path (optional)')
    p.add_argument('--tag', default='')
    args = p.parse_args()

    if args.tasks == 'cw10':
        tasks = CW10_TASKS
    elif args.tasks == 'cw20':
        tasks = CW10_TASKS * 2
    else:
        tasks = ['hammer-v3', 'push-v3', 'window-close-v3', 'faucet-close-v3']

    print(f'Method={args.method} tasks={len(tasks)} steps_per_task={args.steps_per_task} '
          f'seed={args.seed} cl_reg_coef={args.cl_reg_coef}', flush=True)
    print(f'CSC: gamma_comp={args.gamma_comp} beta={args.grad_scale_beta}', flush=True)

    out_path = (f'checkpoints/cw_{args.method}_{args.tasks}_s{args.seed}'
                f'{args.tag}.pt')
    os.makedirs('checkpoints', exist_ok=True)
    ckpt = args.ckpt if args.ckpt else out_path

    results = train(args.method, tasks, args.steps_per_task, args.seed,
                    args.cl_reg_coef, args.replay_ratio, args.replay_per_task,
                    args.gamma_comp, args.grad_scale_beta,
                    args.lr_quant, args.bit_floor, args.reset_alpha,
                    ckpt_path=ckpt, eval_every=args.eval_every)

    final_path = out_path.replace('.pt', '_final.json')
    with open(final_path, 'w') as f:
        json.dump({'config': vars(args),
                   'final': results['final'],
                   'avg': results['avg'],
                   'eval_history': results['eval_history']}, f, indent=2)
    print(f'Saved final results: {final_path}', flush=True)


if __name__ == '__main__':
    main()
