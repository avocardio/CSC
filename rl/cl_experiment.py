"""Continual Learning experiment runner on GPU CW tasks.

Runs a sequence of tasks, training SAC with various CL methods:
- finetune (no CL, baseline)
- ewc
- mas
- replay
- compression (ours)
- csc (ours full)

Reports the standard continual RL metrics:
- Average performance (final success across all tasks)
- Forward transfer (AUC of CL learning curve vs single-task baseline)
- Backward transfer / forgetting (performance drop on old tasks)

Everything runs on GPU — target 10k+ sps per task.
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import time
import argparse
import json

from rl.cw_tasks import make_env, CW_TASK_REGISTRY
from rl.cw_gpu_env import OBS_DIM, ACT_DIM, MAX_EP_LEN


LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0
torch.set_float32_matmul_precision('high')  # TF32 on GH200 — ~2x matmul speedup


# ========================================================================
# Replay buffer + task replay store
# ========================================================================
class ReplayBuffer:
    def __init__(self, capacity, device='cuda:0'):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, OBS_DIM, device=device)
        self.act = torch.zeros(capacity, ACT_DIM, device=device)
        self.rew = torch.zeros(capacity, 1, device=device)
        self.next_obs = torch.zeros(capacity, OBS_DIM, device=device)
        self.done = torch.zeros(capacity, 1, device=device)
        self.pos = 0
        self.size = 0

    def add_batch(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        if self.pos + n <= self.capacity:
            idx = slice(self.pos, self.pos + n)
        else:
            idx = (torch.arange(self.pos, self.pos + n, device=self.device)
                   % self.capacity)
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew.unsqueeze(-1) if rew.dim() == 1 else rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done.unsqueeze(-1).float() if done.dim() == 1 else done.float()
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]

    def snapshot(self, n):
        n = min(n, self.size)
        idx = torch.randperm(self.size, device=self.device)[:n]
        return (self.obs[idx].clone(), self.act[idx].clone(), self.rew[idx].clone(),
                self.next_obs[idx].clone(), self.done[idx].clone())

    def clear(self):
        self.pos = 0
        self.size = 0


class TaskReplayStore:
    """Per-task replay data from completed tasks."""
    def __init__(self, device='cuda:0'):
        self.device = device
        self.data = []

    def add(self, buffer, n=5000):
        self.data.append(buffer.snapshot(n))

    def sample(self, batch_size):
        if not self.data:
            return None
        per_task = max(1, batch_size // len(self.data))
        out = [[] for _ in range(5)]
        for d in self.data:
            n = d[0].shape[0]
            idx = torch.randint(0, n, (per_task,), device=self.device)
            for k in range(5):
                out[k].append(d[k][idx])
        return tuple(torch.cat(x) for x in out)

    @property
    def n_tasks(self):
        return len(self.data)


# ========================================================================
# SAC networks with optional compression importance
# ========================================================================
class Actor(nn.Module):
    def __init__(self, hidden=256, use_compression=False):
        super().__init__()
        self.fc1 = nn.Linear(OBS_DIM, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.ln2 = nn.LayerNorm(hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.ln3 = nn.LayerNorm(hidden)
        self.mean = nn.Linear(hidden, ACT_DIM)
        self.log_std = nn.Linear(hidden, ACT_DIM)
        self.use_compression = use_compression
        if use_compression:
            # Per-unit bit-depth (importance): one per hidden unit
            self.importance = nn.ParameterList([
                nn.Parameter(torch.full((hidden,), 8.0)),
                nn.Parameter(torch.full((hidden,), 8.0)),
                nn.Parameter(torch.full((hidden,), 8.0)),
            ])

    def forward(self, obs):
        h = F.leaky_relu(self.ln1(self.fc1(obs)))
        h = F.leaky_relu(self.ln2(self.fc2(h)))
        h = F.leaky_relu(self.ln3(self.fc3(h)))
        mu = self.mean(h)
        ls = torch.tanh(self.log_std(h))
        ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (ls + 1)
        return mu, ls

    def sample(self, obs):
        mu, ls = self.forward(obs)
        std = ls.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        lp = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, lp

    @torch.no_grad()
    def act(self, obs):
        action, _ = self.sample(obs)
        return action

    @torch.no_grad()
    def act_deterministic(self, obs):
        mu, _ = self.forward(obs)
        return torch.tanh(mu)

    def compression_loss(self):
        if not self.use_compression:
            return torch.tensor(0.0)
        return sum(b.clamp(min=0).mean() for b in self.importance) / len(self.importance)


class Critic(nn.Module):
    def __init__(self, hidden=256):
        super().__init__()
        d = OBS_DIM + ACT_DIM
        self.q1 = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(
            nn.Linear(d, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)


# ========================================================================
# SAC Agent with CL methods
# ========================================================================
class SACAgentCL:
    def __init__(self, method='finetune', lr=1e-3, gamma=0.99, tau=0.005,
                 batch_size=128, replay_ratio=0.25, cl_reg_coef=100.0,
                 gamma_comp=0.01, grad_scale_beta=1.0,
                 device='cuda:0'):
        self.device = device
        self.method = method
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.replay_ratio = replay_ratio
        self.cl_reg_coef = cl_reg_coef
        self.gamma_comp = gamma_comp
        self.grad_scale_beta = grad_scale_beta
        self.use_compression = method in ('compression', 'csc')
        self.use_replay = method in ('replay', 'csc')
        self.use_ewc = method == 'ewc'
        self.use_l2 = method == 'l2'
        self.use_mas = method == 'mas'
        self.use_packnet = method == 'packnet'
        self._current_task = 0

        self.actor = Actor(use_compression=self.use_compression).to(device)
        self.critic = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        # Compile for kernel fusion (CUDA graphs don't work — modules called 2x per step)
        if not self.use_packnet:
            self.actor = torch.compile(self.actor)
            self.critic = torch.compile(self.critic)
            self.critic_target = torch.compile(self.critic_target)

        actor_params = [p for n, p in self.actor.named_parameters()
                        if 'importance' not in n]
        self.actor_opt = torch.optim.Adam(actor_params, lr=lr, fused=True)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr, fused=True)
        if self.use_compression:
            self.imp_opt = torch.optim.Adam(self.actor.importance.parameters(), lr=0.01, fused=True)

        self.target_entropy = -float(ACT_DIM)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.replay_store = TaskReplayStore(device) if self.use_replay else None
        if self.use_compression:
            self.accumulated_importance = [
                torch.zeros(256, device=device) for _ in range(3)
            ]
        if self.use_ewc:
            self.ewc_fisher = {}
            self.ewc_params = {}
        if self.use_l2 or self.use_mas:
            self.reg_params = {}  # snapshot of params from prev task
            self.mas_importance = {}  # MAS: output sensitivity per param
        if self.use_packnet:
            # Integer ownership tensor per kernel parameter (actor only)
            self.pn_owner = {}
            self.pn_freeze_bn = False  # freeze biases/LN after task 0
            for n, p in self.actor.named_parameters():
                if 'weight' in n and p.dim() == 2 and 'importance' not in n:
                    self.pn_owner[n] = torch.zeros_like(p, dtype=torch.int32)
            self.pn_retrain_steps = 100000  # CW paper standard

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def save_checkpoint(self, path, task_idx, eval_matrix, learning_curves):
        """Save full agent state + experiment progress at a task boundary."""
        # Get underlying module if compiled
        actor_sd = (self.actor._orig_mod.state_dict()
                    if hasattr(self.actor, '_orig_mod') else self.actor.state_dict())
        critic_sd = (self.critic._orig_mod.state_dict()
                     if hasattr(self.critic, '_orig_mod') else self.critic.state_dict())
        critic_t_sd = (self.critic_target._orig_mod.state_dict()
                       if hasattr(self.critic_target, '_orig_mod') else
                       self.critic_target.state_dict())
        ckpt = {
            'task_idx': task_idx,
            'method': self.method,
            'actor': actor_sd,
            'critic': critic_sd,
            'critic_target': critic_t_sd,
            'actor_opt': self.actor_opt.state_dict(),
            'critic_opt': self.critic_opt.state_dict(),
            'log_alpha': self.log_alpha.data.clone(),
            'alpha_opt': self.alpha_opt.state_dict(),
            'eval_matrix': eval_matrix,
            'learning_curves': learning_curves,
        }
        if self.use_compression:
            ckpt['imp_opt'] = self.imp_opt.state_dict()
            ckpt['accumulated_importance'] = [a.clone() for a in self.accumulated_importance]
        if self.use_replay and self.replay_store:
            ckpt['replay_data'] = [(d[0].cpu(), d[1].cpu(), d[2].cpu(),
                                    d[3].cpu(), d[4].cpu()) for d in self.replay_store.data]
        if self.use_ewc:
            ckpt['ewc_fisher'] = {n: v.cpu() for n, v in self.ewc_fisher.items()}
            ckpt['ewc_params'] = {n: v.cpu() for n, v in self.ewc_params.items()}
        if self.use_l2 or self.use_mas:
            ckpt['reg_params'] = {n: v.cpu() for n, v in self.reg_params.items()}
        if self.use_mas:
            ckpt['mas_importance'] = {n: v.cpu() for n, v in self.mas_importance.items()}
        if self.use_packnet:
            ckpt['pn_owner'] = {n: v.cpu() for n, v in self.pn_owner.items()}
            ckpt['pn_freeze_bn'] = self.pn_freeze_bn
        torch.save(ckpt, path)
        print(f'  Checkpoint saved: {path} (after task {task_idx})', flush=True)

    def load_checkpoint(self, path):
        """Load agent state from checkpoint. Returns (task_idx, eval_matrix, learning_curves)."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        # Load into underlying module if compiled
        actor_mod = (self.actor._orig_mod if hasattr(self.actor, '_orig_mod')
                     else self.actor)
        critic_mod = (self.critic._orig_mod if hasattr(self.critic, '_orig_mod')
                      else self.critic)
        critic_t_mod = (self.critic_target._orig_mod
                        if hasattr(self.critic_target, '_orig_mod')
                        else self.critic_target)
        actor_mod.load_state_dict(ckpt['actor'])
        critic_mod.load_state_dict(ckpt['critic'])
        critic_t_mod.load_state_dict(ckpt['critic_target'])
        self.actor_opt.load_state_dict(ckpt['actor_opt'])
        self.critic_opt.load_state_dict(ckpt['critic_opt'])
        self.log_alpha.data.copy_(ckpt['log_alpha'])
        self.alpha_opt.load_state_dict(ckpt['alpha_opt'])
        if self.use_compression and 'imp_opt' in ckpt:
            self.imp_opt.load_state_dict(ckpt['imp_opt'])
            self.accumulated_importance = [a.to(self.device) for a in ckpt['accumulated_importance']]
        if self.use_replay and 'replay_data' in ckpt:
            self.replay_store.data = [
                tuple(t.to(self.device) for t in d) for d in ckpt['replay_data']]
        if self.use_ewc and 'ewc_fisher' in ckpt:
            self.ewc_fisher = {n: v.to(self.device) for n, v in ckpt['ewc_fisher'].items()}
            self.ewc_params = {n: v.to(self.device) for n, v in ckpt['ewc_params'].items()}
        if (self.use_l2 or self.use_mas) and 'reg_params' in ckpt:
            self.reg_params = {n: v.to(self.device) for n, v in ckpt['reg_params'].items()}
        if self.use_mas and 'mas_importance' in ckpt:
            self.mas_importance = {n: v.to(self.device) for n, v in ckpt['mas_importance'].items()}
        if self.use_packnet and 'pn_owner' in ckpt:
            self.pn_owner = {n: v.to(self.device) for n, v in ckpt['pn_owner'].items()}
            self.pn_freeze_bn = ckpt['pn_freeze_bn']
        self._current_task = ckpt['task_idx'] + 1
        print(f'  Loaded checkpoint: {path} (resuming from task {self._current_task})')
        return ckpt['task_idx'], ckpt['eval_matrix'], ckpt['learning_curves']

    def update(self, buffer):
        # Sample batch (possibly mixed with replay)
        if self.replay_store and self.replay_store.n_tasks > 0:
            n_curr = int(self.batch_size * (1 - self.replay_ratio))
            n_rep = self.batch_size - n_curr
            s_c, a_c, r_c, ns_c, d_c = buffer.sample(n_curr)
            rep = self.replay_store.sample(n_rep)
            s = torch.cat([s_c, rep[0]])
            a = torch.cat([a_c, rep[1]])
            r = torch.cat([r_c, rep[2]])
            ns = torch.cat([ns_c, rep[3]])
            d = torch.cat([d_c, rep[4]])
        else:
            s, a, r, ns, d = buffer.sample(self.batch_size)

        alpha = self.log_alpha.exp()

        # Critic
        with torch.no_grad():
            na, nlp = self.actor.sample(ns)
            q1t, q2t = self.critic_target(ns, na)
            q1t, q2t = q1t.clone(), q2t.clone()
            qt = r + (1 - d) * self.gamma * (torch.min(q1t, q2t) - alpha * nlp)
        q1, q2 = self.critic(s, a)
        q1, q2 = q1.clone(), q2.clone()  # prevent CUDA graph buffer overwrite
        cl = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
        self.critic_opt.zero_grad(set_to_none=True)
        cl.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        # Actor
        na2, lp2 = self.actor.sample(s)
        q1n, q2n = self.critic(s, na2)
        al = (alpha.detach() * lp2 - torch.min(q1n, q2n)).mean()
        if self.use_compression:
            al = al + self.gamma_comp * self.actor.compression_loss()
        if self.use_ewc and self.ewc_fisher:
            ewc_loss = sum(
                (self.ewc_fisher[n] * (p - self.ewc_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.ewc_fisher
            )
            al = al + self.cl_reg_coef * ewc_loss
        if self.use_l2 and self.reg_params:
            l2_loss = sum(
                (p - self.reg_params[n]).pow(2).sum()
                for n, p in self.actor.named_parameters() if n in self.reg_params
            )
            al = al + self.cl_reg_coef * l2_loss
        if self.use_mas and self.reg_params:
            mas_loss = sum(
                (self.mas_importance[n] * (p - self.reg_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.mas_importance
            )
            al = al + self.cl_reg_coef * mas_loss
        self.actor_opt.zero_grad(set_to_none=True)
        if self.use_compression:
            self.imp_opt.zero_grad()
        al.backward()
        # Gradient scaling by accumulated importance (soft protection)
        if self.use_compression and self.grad_scale_beta > 0:
            self._scale_actor_grads()
        # PackNet: zero gradients for weights owned by previous tasks
        if self.use_packnet:
            self._packnet_mask_grads()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()
        if self.use_compression:
            self.imp_opt.step()

        # Alpha
        alpha_loss = -(self.log_alpha.exp() * (lp2 + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad(set_to_none=True)
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target update
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, self.tau)

    def _scale_actor_grads(self):
        beta = self.grad_scale_beta
        layers = [self.actor.fc1, self.actor.fc2, self.actor.fc3]
        for i, layer in enumerate(layers):
            acc = self.accumulated_importance[i]
            scale = 1.0 / (1.0 + beta * acc)
            if layer.weight.grad is not None:
                layer.weight.grad *= scale.unsqueeze(1)
            if layer.bias.grad is not None:
                layer.bias.grad *= scale

    def _packnet_mask_grads(self):
        """Zero gradients for weights not owned by current task."""
        if not self.use_packnet:
            return
        for n, p in self.actor.named_parameters():
            if p.grad is None:
                continue
            if n in self.pn_owner:
                # Only allow gradient for weights owned by current task
                mask = (self.pn_owner[n] == self._current_task).float()
                p.grad *= mask
            elif self.pn_freeze_bn and ('bias' in n or 'ln' in n):
                # Freeze biases and LayerNorm after task 0
                p.grad.zero_()

    def _packnet_prune(self, task_idx, n_tasks):
        """Prune current task's weights by magnitude, free capacity for future."""
        if not self.use_packnet:
            return
        tasks_left = n_tasks - task_idx - 1
        if tasks_left <= 0:
            return  # last task, no pruning needed
        prune_perc = tasks_left / (tasks_left + 1)

        with torch.no_grad():
            for n, p in self.actor.named_parameters():
                if n not in self.pn_owner:
                    continue
                owner = self.pn_owner[n]
                # Get values owned by current task
                mask = (owner == task_idx)
                vals = p[mask].abs()
                if vals.numel() == 0:
                    continue
                # Find threshold
                k = int(vals.numel() * prune_perc)
                if k == 0:
                    continue
                threshold = vals.sort()[0][k]
                # Zero out pruned weights and transfer ownership
                prune_mask = mask & (p.abs() <= threshold)
                p[prune_mask] = 0.0
                owner[prune_mask] = task_idx + 1  # give to next task

        # Freeze biases/LN after task 0
        if task_idx == 0:
            self.pn_freeze_bn = True

        print(f'    PackNet: pruned {prune_perc:.1%} of task {task_idx} weights')

    def _packnet_retrain(self, buffer):
        """Retrain after pruning with frozen pruned weights."""
        if not self.use_packnet:
            return
        # Reset optimizer
        self.actor_opt = torch.optim.Adam(
            [p for n, p in self.actor.named_parameters() if 'importance' not in n],
            lr=1e-3)
        for _ in range(self.pn_retrain_steps):
            self.update(buffer)
        # Reset optimizer again
        self.actor_opt = torch.optim.Adam(
            [p for n, p in self.actor.named_parameters() if 'importance' not in n],
            lr=1e-3)
        print(f'    PackNet: retrained for {self.pn_retrain_steps} steps')

    def on_task_end(self, buffer, task_idx=0, n_tasks=4):
        """Called after each task. Snapshot state for CL methods."""
        if self.use_replay:
            self.replay_store.add(buffer, n=5000)
        if self.use_compression:
            with torch.no_grad():
                for i, imp in enumerate(self.actor.importance):
                    self.accumulated_importance[i] = torch.max(
                        self.accumulated_importance[i], imp.data.clamp(min=0))
        if self.use_ewc:
            self._compute_ewc_fisher(buffer)
        if self.use_l2:
            # Snapshot params for L2 regularization toward previous task
            for n, p in self.actor.named_parameters():
                if 'importance' not in n:
                    self.reg_params[n] = p.data.clone()
        if self.use_mas:
            self._compute_mas_importance(buffer)
        if self.use_packnet:
            self._packnet_prune(task_idx, n_tasks)
            self._packnet_retrain(buffer)

    def on_task_start(self, task_idx):
        """Called at start of each new task. Reset alpha to encourage exploration."""
        self._current_task = task_idx
        if task_idx > 0:
            # Reset log_alpha to restore exploration
            with torch.no_grad():
                self.log_alpha.fill_(0.0)  # alpha = 1.0
            # Rebuild alpha optimizer to reset its state
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=1e-3)

    def _compute_ewc_fisher(self, buffer, n_batches=10, bs=256):
        params = [(n, p) for n, p in self.actor.named_parameters()
                  if 'importance' not in n and 'mean' not in n and 'log_std' not in n]
        for n, p in params:
            self.ewc_params[n] = p.data.clone()

        fisher = {n: torch.zeros_like(p) for n, p in params}
        for _ in range(n_batches):
            s, _, _, _, _ = buffer.sample(bs)
            mu, ls = self.actor(s)
            dist = Normal(mu, ls.exp())
            x = dist.rsample()
            lp = (dist.log_prob(x) -
                  torch.log(1 - torch.tanh(x).pow(2) + 1e-6)).sum(-1)
            self.actor.zero_grad()
            lp.mean().backward()
            for n, p in params:
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)

        for n in fisher:
            f = (fisher[n] / n_batches).clamp(min=1e-5)
            if n in self.ewc_fisher:
                self.ewc_fisher[n] = self.ewc_fisher[n] + f
            else:
                self.ewc_fisher[n] = f

    def _compute_mas_importance(self, buffer, n_batches=10, bs=256):
        """MAS: importance = mean |grad of actor output w.r.t. params|."""
        params = [(n, p) for n, p in self.actor.named_parameters()
                  if 'importance' not in n and p.requires_grad]
        # Snapshot params
        for n, p in params:
            self.reg_params[n] = p.data.clone()

        importance = {n: torch.zeros_like(p) for n, p in params}
        for _ in range(n_batches):
            s, _, _, _, _ = buffer.sample(bs)
            mu, _ = self.actor(s)
            # MAS uses the L2 norm of the output as the loss
            loss = mu.pow(2).sum()
            self.actor.zero_grad()
            loss.backward()
            for n, p in params:
                if p.grad is not None:
                    importance[n] += p.grad.data.abs()

        for n in importance:
            imp = (importance[n] / n_batches).clamp(min=1e-5)
            if n in self.mas_importance:
                self.mas_importance[n] = self.mas_importance[n] + imp
            else:
                self.mas_importance[n] = imp


# ========================================================================
# Evaluate on a task (deterministic)
# ========================================================================
@torch.no_grad()
def evaluate_task(agent, task_name, n_eval_envs=16, stochastic=True):
    """Evaluate on task using stochastic policy (CW paper convention).

    Runs n_eval_envs episodes of MAX_EP_LEN steps each.
    Returns fraction of episodes with at least one success.
    """
    eval_env = make_env(task_name, n_envs=n_eval_envs)
    obs = eval_env.reset()
    success = torch.zeros(n_eval_envs, device=agent.device)
    for _ in range(MAX_EP_LEN):
        if stochastic:
            action = agent.actor.act(obs)  # stochastic (CW paper default)
        else:
            action = agent.actor.act_deterministic(obs)
        obs, _, done, info = eval_env.auto_reset_step(action)
        success = torch.max(success, info['success_once'].float())
    result = success.mean().item()
    del eval_env
    torch.cuda.empty_cache()
    return result


# ========================================================================
# Main training loop
# ========================================================================
def train_cl(method, tasks, steps_per_task=1_000_000, n_envs=256,
             start_steps=10000, update_freq=1.0, batch_size=128, seed=42,
             ckpt_dir='checkpoints', resume_from=None, run_name=None,
             gamma_comp=0.01, grad_scale_beta=1.0, cl_reg_coef=100.0):
    """Run continual learning training over a sequence of tasks.

    Checkpoints after each task boundary. Resume with resume_from=path.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SACAgentCL(method=method, batch_size=batch_size,
                       gamma_comp=gamma_comp, grad_scale_beta=grad_scale_beta,
                       cl_reg_coef=cl_reg_coef)
    buffer = ReplayBuffer(capacity=1_000_000)

    n = len(tasks)
    eval_matrix = np.zeros((n, n))
    learning_curves = {t: [] for t in tasks}

    start_task = 0
    if resume_from and os.path.exists(resume_from):
        last_idx, eval_matrix, learning_curves = agent.load_checkpoint(resume_from)
        eval_matrix = np.array(eval_matrix)
        start_task = last_idx + 1
        print(f'Resuming from task {start_task}/{n}')

    os.makedirs(ckpt_dir, exist_ok=True)
    t_start = time.time()

    for task_idx in range(start_task, n):
        task_name = tasks[task_idx]
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}/{n-1}: {task_name}")
        print(f"{'='*60}", flush=True)

        env = make_env(task_name, n_envs=n_envs)
        obs = env.reset()
        buffer.clear()
        agent.on_task_start(task_idx)

        task_t0 = time.time()
        task_steps = 0

        while task_steps < steps_per_task:
            if task_steps < start_steps:
                action = torch.rand(n_envs, ACT_DIM, device='cuda:0') * 2 - 1
            else:
                action = agent.actor.act(obs)

            next_obs, reward, done, info = env.auto_reset_step(action)
            buffer.add_batch(obs, action, reward, info['real_next_obs'], done)
            obs = next_obs
            task_steps += n_envs

            if task_steps >= start_steps and buffer.size >= agent.batch_size:
                n_updates = max(1, int(n_envs * update_freq))
                for _ in range(n_updates):
                    agent.update(buffer)

            if task_steps % (n_envs * 40) < n_envs:
                elapsed = time.time() - task_t0
                sps = task_steps / max(elapsed, 1e-6)
                curr_succ = info['success_once'].float().mean().item()
                learning_curves[task_name].append((task_idx, task_steps, curr_succ))
                print(f'  [{task_steps//1000:4d}K/{steps_per_task//1000}K] '
                      f'sps={sps:.0f} succ_curr={curr_succ:.3f} '
                      f'alpha={agent.alpha:.3f}', flush=True)

        agent.on_task_end(buffer, task_idx=task_idx, n_tasks=n)

        print(f'  Evaluating on all {n} tasks...', flush=True)
        for eval_idx, eval_task in enumerate(tasks):
            succ = evaluate_task(agent, eval_task, n_eval_envs=16)
            eval_matrix[task_idx, eval_idx] = succ
            print(f'    {eval_task:20s}: {succ:.3f}', flush=True)

        env.close()

        # Checkpoint after each task
        tag = run_name or f'{method}_s{seed}'
        ckpt_name = f'{tag}_task{task_idx}.pt'
        agent.save_checkpoint(
            os.path.join(ckpt_dir, ckpt_name), task_idx,
            eval_matrix.tolist(), learning_curves)

    total_time = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"DONE in {total_time:.0f}s ({total_time/60:.1f} min)")
    print(f"{'='*60}", flush=True)

    # Compute metrics
    metrics = compute_cl_metrics(eval_matrix, tasks)
    print(f"\nCONTINUAL LEARNING METRICS:")
    for k, v in metrics.items():
        print(f"  {k}: {v:.4f}")

    return {
        'method': method,
        'tasks': tasks,
        'eval_matrix': eval_matrix.tolist(),
        'learning_curves': learning_curves,
        'metrics': metrics,
        'total_time': total_time,
    }


def compute_cl_metrics(eval_matrix: np.ndarray, tasks: list) -> dict:
    """Compute standard CL metrics from eval matrix.

    eval_matrix[i, j] = success rate of task j after training on tasks 0..i.

    Metrics:
        avg_performance: mean of eval_matrix[n-1, :] (final mean across all tasks)
        avg_forgetting: mean_i(eval_matrix[i,i] - eval_matrix[n-1,i]) for i<n-1
                        (how much each task degraded from right after training
                         it to the end)
        backward_transfer: mean_i(eval_matrix[n-1,i] - eval_matrix[i,i]) for i<n-1
                           (negative = forgetting, positive = gain from later tasks)
        forward_transfer: mean_j(eval_matrix[j-1,j]) for j>0
                          (how good we were on task j BEFORE training it,
                           i.e., transfer from earlier tasks)
    """
    n = len(tasks)
    final = eval_matrix[n - 1]  # last row
    avg_performance = final.mean()

    # Forgetting: for each task i < n-1, how much did the post-training performance drop?
    forgetting = []
    for i in range(n - 1):
        drop = eval_matrix[i, i] - eval_matrix[n - 1, i]
        forgetting.append(max(0, drop))
    avg_forgetting = float(np.mean(forgetting)) if forgetting else 0.0

    # Backward transfer: final - right_after (positive = later tasks helped)
    bwt = []
    for i in range(n - 1):
        bwt.append(eval_matrix[n - 1, i] - eval_matrix[i, i])
    avg_bwt = float(np.mean(bwt)) if bwt else 0.0

    # Forward transfer: for task j > 0, eval before training it (row j-1)
    fwt = []
    for j in range(1, n):
        fwt.append(eval_matrix[j - 1, j])
    avg_fwt = float(np.mean(fwt)) if fwt else 0.0

    return {
        'avg_performance': float(avg_performance),
        'avg_forgetting': avg_forgetting,
        'backward_transfer': avg_bwt,
        'forward_transfer': avg_fwt,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='finetune',
                        choices=['finetune', 'l2', 'ewc', 'mas', 'replay',
                                 'compression', 'csc', 'packnet'])
    parser.add_argument('--tasks', type=str, default='reach_cycle',
                        help='Task sequence: reach_cycle, cw10, or comma-separated list')
    parser.add_argument('--steps_per_task', type=int, default=1_000_000)
    parser.add_argument('--n_envs', type=int, default=256)
    parser.add_argument('--batch_size', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='cl_result.json')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--run_name', type=str, default=None,
                        help='Name for checkpoint files (default: method_sSEED)')
    # CSC hyperparameters
    parser.add_argument('--gamma_comp', type=float, default=0.01,
                        help='Compression loss weight (CSC)')
    parser.add_argument('--grad_scale_beta', type=float, default=1.0,
                        help='Gradient scaling strength (CSC)')
    parser.add_argument('--cl_reg_coef', type=float, default=100.0,
                        help='EWC regularization coefficient')
    args = parser.parse_args()

    if args.tasks == 'reach_cycle':
        tasks = ['reach-front', 'reach-top', 'reach-left', 'reach-right']
    elif args.tasks == 'reach_long':
        tasks = ['reach-front', 'reach-top', 'reach-left', 'reach-right',
                 'reach-front', 'reach-top', 'reach-left', 'reach-right']
    elif args.tasks == 'cw_subset':
        tasks = ['push', 'window-close', 'faucet-close', 'handle-press-side']
    elif args.tasks == 'cw_learnable':
        # Tasks that actually learn within budget (no push — needs 200K+)
        tasks = ['window-close', 'faucet-close', 'handle-press-side',
                 'peg-unplug-side']
    elif args.tasks == 'cw_full':
        tasks = ['hammer', 'push-wall', 'faucet-close', 'push-back',
                 'handle-press-side', 'push', 'window-close']
    elif args.tasks == 'cw10':
        tasks = ['hammer', 'push-wall', 'faucet-close', 'push-back', 'stick-pull',
                 'handle-press-side', 'push', 'shelf-place', 'window-close',
                 'peg-unplug-side']
    else:
        tasks = args.tasks.split(',')

    print(f'CL experiment: method={args.method}, tasks={tasks}')
    print(f'steps/task={args.steps_per_task}, n_envs={args.n_envs}, seed={args.seed}')
    if args.resume:
        print(f'Resuming from: {args.resume}')

    result = train_cl(
        method=args.method,
        tasks=tasks,
        steps_per_task=args.steps_per_task,
        n_envs=args.n_envs,
        batch_size=args.batch_size,
        seed=args.seed,
        resume_from=args.resume,
        run_name=args.run_name,
        gamma_comp=args.gamma_comp,
        grad_scale_beta=args.grad_scale_beta,
        cl_reg_coef=args.cl_reg_coef,
    )

    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/{args.out}', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'Saved to checkpoints/{args.out}')


if __name__ == '__main__':
    main()
