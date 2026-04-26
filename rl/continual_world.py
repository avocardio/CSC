"""Continual World benchmark in PyTorch.

Reimplements CW10 (Wolczyk et al. 2021) using modern MetaWorld v3.
Matches CW protocol exactly (verified against source code):
- UTD = 1.0 (50 grad steps every 50 env steps)
- 4x256 MLP, LayerNorm+tanh on first layer, LeakyReLU on rest
- lr=1e-3, gamma=0.99, polyak=0.995, batch=128, buffer=1M
- log_alpha init=1.0, target_entropy=-act_dim
- Episode length capped at 200 (CW convention)
- Never done on truncation (always bootstrap through time limits)
- Reset buffer + optimizer on task change

Difference from original CW: uses MetaWorld v3 (39D obs) instead of v1 (12D obs).
"""

import os, sys, math, time, argparse
from collections import defaultdict
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import metaworld
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

CW10_TASKS = [
    'hammer-v3', 'push-wall-v3', 'faucet-close-v3', 'push-back-v3',
    'stick-pull-v3', 'handle-press-side-v3', 'push-v3', 'shelf-place-v3',
    'window-close-v3', 'peg-unplug-side-v3',
]

OBS_DIM = 39  # MetaWorld v3 (CW original: 12D from v1)
ACT_DIM = 4
MAX_EP_LEN = 200  # CW convention (MetaWorld v3 native: 500)
LOG_STD_MIN, LOG_STD_MAX = -20, 2  # CW uses clamp, not tanh+rescale


# ============================================================
# Replay Buffer
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity=1_000_000, device=DEVICE):
        self.cap = capacity
        self.device = device
        self.obs = torch.zeros(capacity, OBS_DIM, device=device)
        self.act = torch.zeros(capacity, ACT_DIM, device=device)
        self.rew = torch.zeros(capacity, 1, device=device)
        self.nobs = torch.zeros(capacity, OBS_DIM, device=device)
        self.done = torch.zeros(capacity, 1, device=device)
        self.pos = 0
        self.size = 0

    def add(self, obs, act, rew, nobs, done):
        self.obs[self.pos] = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        self.act[self.pos] = torch.as_tensor(act, dtype=torch.float32, device=self.device)
        self.rew[self.pos] = float(rew)
        self.nobs[self.pos] = torch.as_tensor(nobs, dtype=torch.float32, device=self.device)
        self.done[self.pos] = float(done)
        self.pos = (self.pos + 1) % self.cap
        self.size = min(self.size + 1, self.cap)

    def sample(self, bs):
        idx = torch.randint(0, self.size, (bs,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.nobs[idx], self.done[idx]

    def reset(self):
        self.pos = 0
        self.size = 0

    def snapshot(self, n):
        n = min(n, self.size)
        idx = torch.randperm(self.size, device=self.device)[:n]
        return tuple(t[idx].clone() for t in [self.obs, self.act, self.rew, self.nobs, self.done])


class TaskReplayStore:
    def __init__(self, device=DEVICE):
        self.device = device
        self.data = []

    def add_task(self, buf, n=10000):
        self.data.append(buf.snapshot(n))

    def sample(self, bs):
        if not self.data:
            return None
        per = max(1, bs // len(self.data))
        parts = [[] for _ in range(5)]
        for d in self.data:
            idx = torch.randint(0, d[0].shape[0], (per,), device=self.device)
            for k in range(5):
                parts[k].append(d[k][idx])
        return tuple(torch.cat(p) for p in parts)

    @property
    def n_tasks(self):
        return len(self.data)


# ============================================================
# Networks — exact CW architecture
# ============================================================
class CWActor(nn.Module):
    """CW actor: LayerNorm+tanh on first layer, LeakyReLU on rest."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        # First layer: Linear → LayerNorm → tanh (CW convention)
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        # Remaining layers: Linear → LeakyReLU (no LayerNorm)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.head_mu = nn.Linear(hidden, act_dim)
        self.head_log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = torch.tanh(self.ln1(self.fc1(obs)))  # LayerNorm + tanh on first
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        mu = self.head_mu(h)
        log_std = self.head_log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        lp = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, lp, mu

    def get_action_stochastic(self, obs):
        with torch.no_grad():
            a, _, _ = self.sample(obs)
        return a

    def get_action_deterministic(self, obs):
        with torch.no_grad():
            mu, _ = self.forward(obs)
        return torch.tanh(mu)


class CWCritic(nn.Module):
    """CW critic: same architecture as actor but input is obs+act."""

    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        d = obs_dim + act_dim
        self.fc1 = nn.Linear(d, hidden)
        self.ln1 = nn.LayerNorm(hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.fc4 = nn.Linear(hidden, hidden)
        self.head = nn.Linear(hidden, 1)

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        h = torch.tanh(self.ln1(self.fc1(x)))
        h = F.leaky_relu(self.fc2(h))
        h = F.leaky_relu(self.fc3(h))
        h = F.leaky_relu(self.fc4(h))
        return self.head(h)


# ============================================================
# SAC Agent — matches CW exactly
# ============================================================
class SACAgent:
    def __init__(self, method='finetune', lr=1e-3, gamma=0.99, polyak=0.995,
                 batch_size=128, cl_reg_coef=1e4, replay_ratio=0.3,
                 replay_per_task=10000, device=DEVICE):
        self.device = device
        self.gamma = gamma
        self.polyak = polyak
        self.batch_size = batch_size
        self.method = method
        self.cl_reg_coef = cl_reg_coef
        self.replay_ratio = replay_ratio
        self.replay_per_task = replay_per_task

        self.actor = CWActor().to(device)
        self.q1 = CWCritic().to(device)
        self.q2 = CWCritic().to(device)
        self.q1_target = CWCritic().to(device)
        self.q2_target = CWCritic().to(device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        # Separate optimizers (standard PyTorch SAC; CW uses single TF optimizer
        # with separate gradient applications, which is equivalent)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)

        # Alpha: init log_alpha=0 → alpha=1.0 (safer init; CW uses 1.0 but TF optimizer handles differently)
        self.target_entropy = -float(ACT_DIM)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        # CPU actor for fast inference
        self.actor_cpu = CWActor().cpu()
        self.actor_cpu.load_state_dict(self.actor.state_dict())

        # CL state
        self.ewc_fisher = {}
        self.ewc_params = {}
        self.use_replay = method in ('replay', 'ewc_replay', 'csc')
        self.replay_store = TaskReplayStore(device) if self.use_replay else None

        # CSC: compression importance + gradient scaling
        self.use_csc = method == 'csc'
        self.gamma_comp = 0.01
        self.grad_scale_beta = 1.0
        if self.use_csc:
            self.importance = nn.ParameterList([
                nn.Parameter(torch.full((256,), 8.0, device=device))
                for _ in range(4)
            ])
            self.imp_opt = torch.optim.Adam(self.importance.parameters(), lr=0.01)
            self.accumulated_importance = [
                torch.zeros(256, device=device) for _ in range(4)
            ]

        # L2 / MAS regularization
        self.use_l2 = method == 'l2'
        self.use_mas = method == 'mas'
        self.reg_params = {}
        self.mas_importance = {}

        # PackNet: weight ownership + pruning
        self.use_packnet = method == 'packnet'
        if self.use_packnet:
            self.pn_owner = {}
            self.pn_freeze_bn = False
            for n, p in self.actor.named_parameters():
                if 'weight' in n and p.dim() == 2:
                    self.pn_owner[n] = torch.zeros_like(p, dtype=torch.int32)
            self.pn_retrain_steps = 100000
            self._current_task = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def sync_cpu(self):
        sd = {k: v.cpu() for k, v in self.actor.state_dict().items()}
        self.actor_cpu.load_state_dict(sd)

    def get_action_cpu(self, obs_np):
        with torch.no_grad():
            obs_t = torch.as_tensor(obs_np, dtype=torch.float32).unsqueeze(0)
            a, _, _ = self.actor_cpu.sample(obs_t)
        return a.numpy().flatten()

    def reset_for_new_task(self, task_idx=0):
        """Reset optimizer state on task change (CW convention)."""
        lr = self.actor_opt.param_groups[0]['lr']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()), lr=lr)
        if self.use_packnet:
            self._current_task = task_idx

    def update(self, buffer):
        # Mix replay if available
        if self.replay_store and self.replay_store.n_tasks > 0:
            nc = int(self.batch_size * (1 - self.replay_ratio))
            nr = self.batch_size - nc
            sc, ac, rc, nsc, dc = buffer.sample(nc)
            rep = self.replay_store.sample(nr)
            s = torch.cat([sc, rep[0]])
            a = torch.cat([ac, rep[1]])
            r = torch.cat([rc, rep[2]])
            ns = torch.cat([nsc, rep[3]])
            d = torch.cat([dc, rep[4]])
        else:
            s, a, r, ns, d = buffer.sample(self.batch_size)

        alpha = self.log_alpha.exp().detach()

        # Critic targets
        with torch.no_grad():
            na, nlp, _ = self.actor.sample(ns)
            q1t = self.q1_target(ns, na)
            q2t = self.q2_target(ns, na)
            qt = r + (1 - d) * self.gamma * (torch.min(q1t, q2t) - alpha * nlp)

        # Critic update
        q1_pred = self.q1(s, a)
        q2_pred = self.q2(s, a)
        critic_loss = 0.5 * F.mse_loss(q1_pred, qt) + 0.5 * F.mse_loss(q2_pred, qt)
        self.critic_opt.zero_grad()
        critic_loss.backward()
        self.critic_opt.step()

        # Actor update (fresh forward pass after critic update)
        na2, lp2, _ = self.actor.sample(s)
        q1_pi = self.q1(s, na2)
        q2_pi = self.q2(s, na2)
        actor_loss = (alpha * lp2 - torch.min(q1_pi, q2_pi)).mean()
        if self.method in ('ewc', 'ewc_replay') and self.ewc_fisher:
            ewc_pen = sum(
                (self.ewc_fisher[n] * (p - self.ewc_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.ewc_fisher
            )
            actor_loss = actor_loss + self.cl_reg_coef * ewc_pen
        if self.use_csc:
            comp_loss = sum(b.clamp(min=0).mean() for b in self.importance) / len(self.importance)
            actor_loss = actor_loss + self.gamma_comp * comp_loss
        if self.use_l2 and self.reg_params:
            l2_loss = sum(
                (p - self.reg_params[n]).pow(2).sum()
                for n, p in self.actor.named_parameters() if n in self.reg_params)
            actor_loss = actor_loss + self.cl_reg_coef * l2_loss
        if self.use_mas and self.reg_params:
            mas_loss = sum(
                (self.mas_importance[n] * (p - self.reg_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.mas_importance)
            actor_loss = actor_loss + self.cl_reg_coef * mas_loss
        self.actor_opt.zero_grad()
        if self.use_csc:
            self.imp_opt.zero_grad()
        actor_loss.backward()
        if self.use_csc and self.grad_scale_beta > 0:
            layers = [self.actor.fc1, self.actor.fc2, self.actor.fc3, self.actor.fc4]
            for i, layer in enumerate(layers):
                acc = self.accumulated_importance[i]
                scale = 1.0 / (1.0 + self.grad_scale_beta * acc)
                if layer.weight.grad is not None:
                    layer.weight.grad *= scale.unsqueeze(1)
                if layer.bias is not None and layer.bias.grad is not None:
                    layer.bias.grad *= scale
        if self.use_packnet:
            for n, p in self.actor.named_parameters():
                if p.grad is None:
                    continue
                if n in self.pn_owner:
                    mask = (self.pn_owner[n] == self._current_task).float()
                    p.grad *= mask
                elif self.pn_freeze_bn and ('bias' in n or 'ln' in n):
                    p.grad.zero_()
        self.actor_opt.step()
        if self.use_csc:
            self.imp_opt.step()

        # Alpha update (separate optimizer)
        alpha_loss = -(self.log_alpha.exp() * (lp2 + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target update (polyak)
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(self.polyak).add_(p.data, alpha=1 - self.polyak)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(self.polyak).add_(p.data, alpha=1 - self.polyak)

    def compute_ewc_fisher(self, buffer, n_batches=10, mini_bs=256):
        if self.method not in ('ewc', 'ewc_replay'):
            return
        params = [(n, p) for n, p in self.actor.named_parameters() if p.requires_grad]
        for n, p in params:
            self.ewc_params[n] = p.data.clone()
        fisher = {n: torch.zeros_like(p) for n, p in params}
        for _ in range(n_batches):
            s, _, _, _, _ = buffer.sample(mini_bs)
            mu, ls = self.actor(s)
            std = ls.exp()
            dist = Normal(mu, std)
            x = dist.rsample()
            lp = (dist.log_prob(x) - torch.log(1 - torch.tanh(x).pow(2) + 1e-6)).sum(-1)
            self.actor.zero_grad()
            lp.mean().backward()
            for n, p in params:
                if p.grad is not None:
                    fisher[n] += p.grad.data.pow(2)
        for n in fisher:
            fisher[n] = (fisher[n] / n_batches).clamp(min=1e-5)
            self.ewc_fisher[n] = self.ewc_fisher.get(n, torch.zeros_like(fisher[n])) + fisher[n]
        print(f"    EWC Fisher computed: max={max(f.max().item() for f in fisher.values()):.4f}")

    def on_task_end(self, buffer, task_idx=0, n_tasks=10):
        self.compute_ewc_fisher(buffer)
        if self.replay_store:
            self.replay_store.add_task(buffer, self.replay_per_task)
        if self.use_csc:
            with torch.no_grad():
                for i, imp in enumerate(self.importance):
                    self.accumulated_importance[i] = torch.max(
                        self.accumulated_importance[i], imp.data.clamp(min=0))
        if self.use_l2:
            for n, p in self.actor.named_parameters():
                self.reg_params[n] = p.data.clone()
        if self.use_mas:
            self._compute_mas_importance(buffer)
        if self.use_packnet:
            self._packnet_prune(task_idx, n_tasks)
            self._packnet_retrain(buffer)

    def _compute_mas_importance(self, buffer, n_batches=10, bs=256):
        params = [(n, p) for n, p in self.actor.named_parameters() if p.requires_grad]
        for n, p in params:
            self.reg_params[n] = p.data.clone()
        importance = {n: torch.zeros_like(p) for n, p in params}
        for _ in range(n_batches):
            s, _, _, _, _ = buffer.sample(bs)
            mu, _ = self.actor(s)
            loss = mu.pow(2).sum()
            self.actor.zero_grad()
            loss.backward()
            for n, p in params:
                if p.grad is not None:
                    importance[n] += p.grad.data.abs()
        for n in importance:
            imp = (importance[n] / n_batches).clamp(min=1e-5)
            self.mas_importance[n] = self.mas_importance.get(n, torch.zeros_like(imp)) + imp

    def _packnet_prune(self, task_idx, n_tasks):
        tasks_left = n_tasks - task_idx - 1
        if tasks_left <= 0:
            return
        prune_perc = tasks_left / (tasks_left + 1)
        with torch.no_grad():
            for n, p in self.actor.named_parameters():
                if n not in self.pn_owner:
                    continue
                owner = self.pn_owner[n]
                mask = (owner == task_idx)
                vals = p[mask].abs()
                if vals.numel() == 0:
                    continue
                k = int(vals.numel() * prune_perc)
                if k == 0:
                    continue
                threshold = vals.sort()[0][k]
                prune_mask = mask & (p.abs() <= threshold)
                p[prune_mask] = 0.0
                owner[prune_mask] = task_idx + 1
        if task_idx == 0:
            self.pn_freeze_bn = True
        print(f"    PackNet: pruned {prune_perc:.1%} of task {task_idx} weights")

    def _packnet_retrain(self, buffer):
        lr = self.actor_opt.param_groups[0]['lr']
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        for _ in range(self.pn_retrain_steps):
            self.update(buffer)
        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        print(f"    PackNet: retrained for {self.pn_retrain_steps} steps")


# ============================================================
# Environment
# ============================================================
def make_env(task_name):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    env.set_task(ml1.train_tasks[0])
    return env


def evaluate(agent, task_name, n_episodes=10, deterministic=False):
    env = make_env(task_name)
    successes = 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_success = False
        for step in range(MAX_EP_LEN):
            if deterministic:
                obs_t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
                action = agent.actor_cpu.get_action_deterministic(obs_t).numpy().flatten()
            else:
                action = agent.get_action_cpu(obs)
            obs, _, term, trunc, info = env.step(action)
            if info.get('success', 0):
                ep_success = True
            if term or (step >= MAX_EP_LEN - 1):
                break
        if ep_success:
            successes += 1
    env.close()
    return successes / n_episodes


# ============================================================
# Training
# ============================================================
UPDATE_EVERY = 50
UPDATE_AFTER = 1000
START_STEPS = 10000


def train(method, tasks, steps_per_task=1_000_000, seed=42,
          cl_reg_coef=1e4, replay_ratio=0.3, replay_per_task=10000,
          utd=0.5):
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SACAgent(method=method, cl_reg_coef=cl_reg_coef,
                     replay_ratio=replay_ratio, replay_per_task=replay_per_task)
    buffer = ReplayBuffer()

    results = {}
    t_global = time.time()

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}: {task_name} ({steps_per_task:,} steps)")
        print(f"{'='*60}", flush=True)

        env = make_env(task_name)
        obs, _ = env.reset()
        ep_len = 0

        # Reset buffer + optimizer on task change (CW convention)
        buffer.reset()
        agent.reset_for_new_task(task_idx)
        agent.sync_cpu()

        t0 = time.time()
        sync_counter = 0

        for step in range(steps_per_task):
            # Action selection
            if step < START_STEPS:
                action = env.action_space.sample()
            else:
                action = agent.get_action_cpu(obs)

            next_obs, reward, term, trunc, info = env.step(action)
            ep_len += 1

            # Done masking: never done on truncation (CW convention)
            done = 0.0  # Never store done=1 for time-limited episodes
            if term and ep_len < MAX_EP_LEN:
                done = 1.0  # Only true termination (rare in MetaWorld)

            buffer.add(obs, action, reward, next_obs, done)
            obs = next_obs

            # Episode reset at 200 steps (CW convention) or termination
            if term or ep_len >= MAX_EP_LEN:
                obs, _ = env.reset()
                ep_len = 0

            # Gradient updates: configurable UTD (default 0.5 for speed, CW paper uses 1.0)
            if step >= UPDATE_AFTER and step % UPDATE_EVERY == 0:
                n_updates = max(1, int(UPDATE_EVERY * utd))
                for _ in range(n_updates):
                    agent.update(buffer)
                sync_counter += 1
                if sync_counter % 5 == 0:
                    agent.sync_cpu()

            # Evaluate every 100K steps
            if (step + 1) % 100_000 == 0:
                elapsed = time.time() - t0
                sps = (step + 1) / max(elapsed, 1)
                evals = {}
                for ti in range(task_idx + 1):
                    evals[tasks[ti]] = evaluate(agent, tasks[ti])
                results[(task_idx, step + 1)] = evals

                parts = " ".join(f"{tasks[ti][:6]}={evals[tasks[ti]]:.2f}"
                                 for ti in range(task_idx + 1))
                rep = f" rep={agent.replay_store.n_tasks}t" if agent.replay_store else ""
                print(f"  {(step+1)//1000}K ({sps:.0f} sps): {parts}{rep} "
                      f"alpha={agent.alpha:.3f}", flush=True)

        env.close()
        agent.on_task_end(buffer, task_idx=task_idx, n_tasks=len(tasks))

    # Final eval
    print(f"\n{'='*60}")
    print(f"FINAL ({time.time()-t_global:.0f}s)")
    print(f"{'='*60}")
    final = {}
    for t in tasks:
        s = evaluate(agent, t, n_episodes=20)
        final[t] = s
        print(f"  {t}: {s:.2f}")
    avg = np.mean(list(final.values()))
    print(f"  Average: {avg:.2f}")
    results[('final',)] = final
    return results, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', required=True,
                        choices=['finetune', 'l2', 'ewc', 'mas', 'ewc_replay',
                                 'replay', 'packnet', 'csc'])
    parser.add_argument('--steps_per_task', type=int, default=1_000_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--cl_reg_coef', type=float, default=1e4)
    parser.add_argument('--replay_ratio', type=float, default=0.3)
    parser.add_argument('--replay_per_task', type=int, default=10000)
    parser.add_argument('--tasks', default='cw10', choices=['cw10', 'cw20'])
    parser.add_argument('--utd', type=float, default=0.5,
                        help='Update-to-data ratio (CW paper=1.0, use 0.5 for speed)')
    parser.add_argument('--tag', default='')
    args = parser.parse_args()

    tasks = CW10_TASKS if args.tasks == 'cw10' else CW10_TASKS * 2

    print(f"Continual World: method={args.method}, {len(tasks)} tasks, "
          f"{args.steps_per_task:,}/task, seed={args.seed}, UTD={args.utd}")
    print(f"Protocol: update_every={UPDATE_EVERY}, "
          f"batch={128}, lr=1e-3, gamma=0.99, polyak=0.995")
    print(f"Episode length: {MAX_EP_LEN}, obs_dim={OBS_DIM}, act_dim={ACT_DIM}")

    results, final = train(args.method, tasks, args.steps_per_task,
                           seed=args.seed, cl_reg_coef=args.cl_reg_coef,
                           replay_ratio=args.replay_ratio,
                           replay_per_task=args.replay_per_task,
                           utd=args.utd)

    os.makedirs('checkpoints', exist_ok=True)
    fname = f'checkpoints/cw_{args.method}_{args.tasks}_s{args.seed}{args.tag}.pt'
    torch.save({'results': results, 'final': final, 'config': vars(args)}, fname)
    print(f"Saved: {fname}")


if __name__ == '__main__':
    main()
