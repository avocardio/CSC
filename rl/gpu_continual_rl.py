"""GPU-accelerated continual RL on ManiSkill3 manipulation tasks.

Compares:
- standard: SAC baseline (catastrophic forgetting)
- replay: SAC + experience replay from past tasks
- compression: compression-guided unit replacement + gradient scaling
- compression_replay: full method (compression + replay)
"""

import os
os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json'

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import mani_skill.render.utils
mani_skill.render.utils.can_render = lambda device: False

import mani_skill.envs
import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from mani_skill.vector.wrappers.gymnasium import ManiSkillVectorEnv
import numpy as np
import time
import argparse

LOG_STD_MIN, LOG_STD_MAX = -5, 2


class GPUReplayBuffer:
    def __init__(self, capacity, obs_dim, act_dim, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.rew = torch.zeros(capacity, 1, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.done = torch.zeros(capacity, 1, device=device)

    def add(self, obs, act, rew, next_obs, done):
        n = obs.shape[0]
        idx = torch.arange(self.pos, self.pos + n, device=self.device) % self.capacity
        self.obs[idx] = obs
        self.act[idx] = act
        self.rew[idx] = rew.unsqueeze(-1) if rew.dim() == 1 else rew
        self.next_obs[idx] = next_obs
        self.done[idx] = done.unsqueeze(-1) if done.dim() == 1 else done
        self.pos = (self.pos + n) % self.capacity
        self.size = min(self.size + n, self.capacity)

    def sample(self, batch_size):
        idx = torch.randint(0, self.size, (batch_size,), device=self.device)
        return self.obs[idx], self.act[idx], self.rew[idx], self.next_obs[idx], self.done[idx]

    def sample_flat(self, n):
        """Sample n transitions as a snapshot (for replay store)."""
        n = min(n, self.size)
        idx = torch.randperm(self.size, device=self.device)[:n]
        return (self.obs[idx].clone(), self.act[idx].clone(),
                self.rew[idx].clone(), self.next_obs[idx].clone(),
                self.done[idx].clone())


class TaskReplayStore:
    """Stores fixed-size replay buffers from completed tasks."""

    def __init__(self, device='cuda'):
        self.device = device
        self.task_data = []  # list of (obs, act, rew, next_obs, done) tuples
        self.total_size = 0

    def add_task(self, buffer, n_samples=50_000):
        """Snapshot n_samples from the current buffer for this task."""
        data = buffer.sample_flat(n_samples)
        self.task_data.append(data)
        self.total_size += data[0].shape[0]
        print(f"    Replay store: saved {data[0].shape[0]} transitions "
              f"(total {self.total_size} across {len(self.task_data)} tasks)")

    def sample(self, batch_size):
        """Sample uniformly across all stored tasks."""
        if not self.task_data:
            return None
        # Split batch evenly across tasks
        per_task = max(1, batch_size // len(self.task_data))
        batches = {k: [] for k in ['s', 'a', 'r', 'ns', 'd']}
        for obs, act, rew, nobs, done in self.task_data:
            n = obs.shape[0]
            idx = torch.randint(0, n, (per_task,), device=self.device)
            batches['s'].append(obs[idx])
            batches['a'].append(act[idx])
            batches['r'].append(rew[idx])
            batches['ns'].append(nobs[idx])
            batches['d'].append(done[idx])
        return (torch.cat(batches['s']), torch.cat(batches['a']),
                torch.cat(batches['r']), torch.cat(batches['ns']),
                torch.cat(batches['d']))

    @property
    def n_tasks(self):
        return len(self.task_data)


class SACActor(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        self.fc1 = nn.Linear(obs_dim, hidden)
        self.fc2 = nn.Linear(hidden, hidden)
        self.fc3 = nn.Linear(hidden, hidden)
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)
        self.importance = nn.ParameterList([
            nn.Parameter(torch.full((hidden,), 8.0)),
            nn.Parameter(torch.full((hidden,), 8.0)),
            nn.Parameter(torch.full((hidden,), 8.0)),
        ])

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
        h = F.relu(self.fc3(h))
        mu = self.mean(h)
        ls = torch.tanh(self.log_std(h))
        ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (ls + 1)
        return mu, ls

    def sample(self, obs):
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        dist = Normal(mu, std)
        x = dist.rsample()
        action = torch.tanh(x)
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    def get_eval_action(self, obs):
        mu, _ = self.forward(obs)
        return torch.tanh(mu)

    def compression_loss(self):
        return sum(b.clamp(min=0).mean() for b in self.importance) / len(self.importance)


class SACCritic(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden=256):
        super().__init__()
        d = obs_dim + act_dim
        self.q1 = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))
        self.q2 = nn.Sequential(nn.Linear(d, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, hidden), nn.ReLU(),
                                nn.Linear(hidden, 1))

    def forward(self, obs, act):
        x = torch.cat([obs, act], -1)
        return self.q1(x), self.q2(x)


class GPUSACAgent:
    def __init__(self, obs_dim, act_dim, hidden=256, lr=3e-4,
                 gamma=0.8, tau=0.01, batch_size=1024,
                 training_freq=64, utd=0.5,
                 use_compression=False, gamma_comp=0.01,
                 replacement_rate=0.001, grad_scale_beta=5.0,
                 use_replay=False, replay_ratio=0.2,
                 use_ewc=False, ewc_lambda=10000.0,
                 use_mas=False, mas_lambda=10000.0,
                 device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.training_freq = training_freq
        self.grad_steps = int(training_freq * utd)
        self.use_compression = use_compression
        self.gamma_comp = gamma_comp
        self.replacement_rate = replacement_rate
        self.grad_scale_beta = grad_scale_beta
        self.use_replay = use_replay
        self.replay_ratio = replay_ratio
        self.use_ewc = use_ewc
        self.ewc_lambda = ewc_lambda
        self.use_mas = use_mas
        self.mas_lambda = mas_lambda
        self.act_dim = act_dim

        self.actor = SACActor(obs_dim, act_dim, hidden).to(device)
        self.critic = SACCritic(obs_dim, act_dim, hidden).to(device)
        self.critic_target = SACCritic(obs_dim, act_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        actor_params = [p for n, p in self.actor.named_parameters() if 'importance' not in n]
        self.actor_opt = torch.optim.Adam(actor_params, lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if use_compression:
            self.imp_opt = torch.optim.Adam(self.actor.importance.parameters(), lr=0.01)

        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.update_count = 0
        self.total_replaced = 0
        if use_compression:
            self.accumulated_importance = [
                torch.zeros(hidden, device=device) for _ in range(3)
            ]
        if use_replay:
            self.replay_store = TaskReplayStore(device)
        if use_ewc:
            # Actor-only EWC (Continual World convention)
            # Fisher is accumulated additively across tasks, θ* is the last task's endpoint
            self.ewc_fisher_actor = {}  # name -> tensor (summed over tasks)
            self.ewc_params_actor = {}  # name -> tensor (snapshot at last task end)
        if use_mas:
            # Actor-only MAS (Continual World convention)
            self.mas_omega_actor = {}  # name -> tensor (summed over tasks)
            self.mas_params_actor = {}  # name -> tensor (snapshot at last task end)

    def snapshot_importance(self):
        if not self.use_compression:
            return
        with torch.no_grad():
            for i, imp in enumerate(self.actor.importance):
                self.accumulated_importance[i] = torch.max(
                    self.accumulated_importance[i], imp.data.clamp(min=0)
                )

    def save_task_replay(self, buffer, n_samples=50_000):
        if self.use_replay:
            self.replay_store.add_task(buffer, n_samples)

    def compute_ewc_fisher(self, buffer, mini_bs=32, n_batches=80):
        """Compute Fisher info faithfully to Continual World (Wolczyk et al. 2021).

        - Fisher is from log π(a|s), NOT SAC actor loss (avoids mixing with Q)
        - Actor-only regularization (CW paper shows critic reg HURTS)
        - Uses small batches (n_batches × mini_bs = 2560 samples) to approximate
          per-sample Fisher better than a single large-batch gradient
        - Clips Fisher from below at 1e-5 for numerical stability
        - Additive accumulation across tasks, one shared θ*
        - Snapshots θ* BEFORE computing Fisher (canonical ordering)
        """
        if not self.use_ewc:
            return
        actor_params_list = [(n, p) for n, p in self.actor.named_parameters()
                              if 'importance' not in n and p.requires_grad
                              and 'log_std' not in n.split('.')[0]]  # include log_std weight, not log_alpha
        # Snapshot θ* BEFORE Fisher computation
        for n, p in actor_params_list:
            self.ewc_params_actor[n] = p.data.clone()

        fisher_a = {n: torch.zeros_like(p) for n, p in actor_params_list}

        for _ in range(n_batches):
            s, _, _, _, _ = buffer.sample(mini_bs)
            # Policy Fisher: sample action from current policy, compute log π(a|s)
            mu, log_std = self.actor(s)
            std = log_std.exp()
            dist = Normal(mu, std)
            x = dist.rsample()  # pre-tanh sample
            action = torch.tanh(x)
            log_prob = (dist.log_prob(x) -
                        torch.log(1 - action.pow(2) + 1e-6)).sum(-1)  # per-sample log π
            # Empirical Fisher approximation: square of batch-mean grad
            # (standard in PyTorch EWC ports; smaller batches reduce the bias)
            self.actor.zero_grad()
            log_prob.mean().backward()
            for n, p in actor_params_list:
                if p.grad is not None:
                    fisher_a[n] += p.grad.data.pow(2)

        # Average across batches, clip, accumulate
        for n, _ in actor_params_list:
            fisher_a[n] /= n_batches
            fisher_a[n].clamp_(min=1e-5)
            if n in self.ewc_fisher_actor:
                self.ewc_fisher_actor[n] = self.ewc_fisher_actor[n] + fisher_a[n]
            else:
                self.ewc_fisher_actor[n] = fisher_a[n]

        max_f = max(f.max().item() for f in self.ewc_fisher_actor.values())
        mean_f = sum(f.mean().item() for f in self.ewc_fisher_actor.values()) / len(self.ewc_fisher_actor)
        print(f"    EWC: Fisher computed (actor only). "
              f"max={max_f:.4f}, mean={mean_f:.6f}, n_params_tracked={len(self.ewc_fisher_actor)}")

    def ewc_penalty_actor(self):
        if not self.use_ewc or not self.ewc_fisher_actor:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.actor.named_parameters():
            if n in self.ewc_fisher_actor:
                loss = loss + (self.ewc_fisher_actor[n] *
                               (p - self.ewc_params_actor[n]).pow(2)).sum()
        return loss

    def ewc_penalty_critic(self):
        # Continual World convention: do NOT regularize critic
        return torch.tensor(0.0, device=self.device)

    def compute_mas_omega(self, buffer, mini_bs=256, n_batches=10):
        """Compute MAS importance Omega faithfully to Continual World (Wolczyk 2021).

        - Omega_i = E_s [ |∂(||mu(s)||^2 + ||log_std(s)||^2)/∂theta_i| ]
        - Actor-only (per CW convention)
        - 10 × 256 = 2560 samples from current task buffer
        - Additive accumulation across tasks (sum omegas)
        - Snapshot theta* BEFORE computing Omega
        """
        if not self.use_mas:
            return
        actor_params_list = [(n, p) for n, p in self.actor.named_parameters()
                              if 'importance' not in n and p.requires_grad]
        # Snapshot theta*
        for n, p in actor_params_list:
            self.mas_params_actor[n] = p.data.clone()

        omega = {n: torch.zeros_like(p) for n, p in actor_params_list}

        for _ in range(n_batches):
            s, _, _, _, _ = buffer.sample(mini_bs)
            mu, log_std = self.actor(s)
            # CW: output norm = ||mu||^2 + ||log_std||^2, summed over batch
            output_norm = (mu.pow(2).sum(-1) + log_std.pow(2).sum(-1)).sum()
            self.actor.zero_grad()
            output_norm.backward()
            for n, p in actor_params_list:
                if p.grad is not None:
                    omega[n] += p.grad.data.abs() / mini_bs

        # Average over batches
        for n in omega:
            omega[n] /= n_batches
            if n in self.mas_omega_actor:
                self.mas_omega_actor[n] = self.mas_omega_actor[n] + omega[n]
            else:
                self.mas_omega_actor[n] = omega[n]

        max_o = max(o.max().item() for o in self.mas_omega_actor.values())
        mean_o = sum(o.mean().item() for o in self.mas_omega_actor.values()) / len(self.mas_omega_actor)
        print(f"    MAS: Omega computed (actor only). "
              f"max={max_o:.4f}, mean={mean_o:.6f}, n_params={len(self.mas_omega_actor)}")

    def mas_penalty_actor(self):
        if not self.use_mas or not self.mas_omega_actor:
            return torch.tensor(0.0, device=self.device)
        loss = torch.tensor(0.0, device=self.device)
        for n, p in self.actor.named_parameters():
            if n in self.mas_omega_actor:
                loss = loss + (self.mas_omega_actor[n] *
                               (p - self.mas_params_actor[n]).pow(2)).sum()
        return loss

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def act(self, obs):
        action, _ = self.actor.sample(obs)
        return action

    @torch.no_grad()
    def act_deterministic(self, obs):
        return self.actor.get_eval_action(obs)

    def update(self, buffer):
        if buffer.size < self.batch_size:
            return {}

        has_replay = self.use_replay and self.replay_store.n_tasks > 0

        for _ in range(self.grad_steps):
            # Sample current task data
            if has_replay:
                n_current = int(self.batch_size * (1 - self.replay_ratio))
                n_replay = self.batch_size - n_current
                s_c, a_c, r_c, ns_c, d_c = buffer.sample(n_current)
                replay_data = self.replay_store.sample(n_replay)
                s = torch.cat([s_c, replay_data[0]])
                a = torch.cat([a_c, replay_data[1]])
                r = torch.cat([r_c, replay_data[2]])
                ns = torch.cat([ns_c, replay_data[3]])
                d = torch.cat([d_c, replay_data[4]])
            else:
                s, a, r, ns, d = buffer.sample(self.batch_size)

            alpha = self.log_alpha.exp()

            # Critic update
            with torch.no_grad():
                na, nlp = self.actor.sample(ns)
                q1t, q2t = self.critic_target(ns, na)
                qt = r + (1 - d) * self.gamma * (torch.min(q1t, q2t) - alpha * nlp)

            q1, q2 = self.critic(s, a)
            cl = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
            # Note: EWC does NOT regularize the critic (Continual World convention)
            self.critic_opt.zero_grad()
            cl.backward()
            self.critic_opt.step()

            # Actor update
            na2, lp2 = self.actor.sample(s)
            q1n, q2n = self.critic(s, na2)
            al = (alpha.detach() * lp2 - torch.min(q1n, q2n)).mean()
            if self.use_compression:
                al = al + self.gamma_comp * self.actor.compression_loss()
            if self.use_ewc and self.ewc_fisher_actor:
                al = al + self.ewc_lambda * self.ewc_penalty_actor()
            if self.use_mas and self.mas_omega_actor:
                al = al + self.mas_lambda * self.mas_penalty_actor()
            self.actor_opt.zero_grad()
            if self.use_compression:
                self.imp_opt.zero_grad()
            al.backward()
            if self.use_compression and self.grad_scale_beta > 0:
                self._scale_actor_gradients()
            self.actor_opt.step()
            if self.use_compression:
                self.imp_opt.step()

            # Alpha update
            alpha_loss = -(self.log_alpha.exp() * (lp2 + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # Target update
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, self.tau)

        self.update_count += 1
        if self.use_compression and self.update_count % 50 == 0:
            self._replace_units()
        return {'cl': cl.item(), 'al': al.item()}

    def _scale_actor_gradients(self):
        beta = self.grad_scale_beta
        for i, layer in enumerate([self.actor.fc1, self.actor.fc2, self.actor.fc3]):
            acc_imp = self.accumulated_importance[i]
            scale = 1.0 / (1.0 + beta * acc_imp)
            if layer.weight.grad is not None:
                layer.weight.grad *= scale.unsqueeze(1)
            if layer.bias.grad is not None:
                layer.bias.grad *= scale

    def _replace_units(self):
        with torch.no_grad():
            for i, (layer, imp) in enumerate([
                (self.actor.fc1, self.actor.importance[0]),
                (self.actor.fc2, self.actor.importance[1]),
                (self.actor.fc3, self.actor.importance[2]),
            ]):
                bits = torch.max(imp.clamp(min=0), self.accumulated_importance[i])
                n = bits.shape[0]
                nr = self.replacement_rate * n
                if nr < 1:
                    if torch.rand(1).item() < nr: nr = 1
                    else: continue
                nr = int(nr)
                _, lowest = torch.topk(-bits, nr)
                nn.init.kaiming_normal_(layer.weight.data[lowest])
                layer.bias.data[lowest] = 0
                imp.data[lowest] = 8.0
                self.accumulated_importance[i][lowest] = 8.0
                self.total_replaced += nr


def make_env(task_name, n_envs, ignore_terminations=True):
    raw = gym.make(task_name, num_envs=n_envs, sim_backend='gpu',
                   render_mode=None, obs_mode='state',
                   control_mode='pd_joint_delta_pos')
    return ManiSkillVectorEnv(raw, n_envs, ignore_terminations=ignore_terminations,
                              record_metrics=True)


def evaluate(agent, task_name, obs_dim, n_eval_envs=16, max_steps=200, device='cuda'):
    env = make_env(task_name, n_eval_envs, ignore_terminations=False)
    obs, _ = env.reset()
    obs = pad_obs(obs, obs_dim)
    total_success = 0
    total_eps = 0

    for _ in range(max_steps):
        action = agent.act_deterministic(obs)
        next_obs, reward, term, trunc, info = env.step(action)
        obs = pad_obs(next_obs, obs_dim)
        if "final_info" in info:
            mask = info["_final_info"]
            n_done = mask.sum().item()
            if n_done > 0:
                ep_info = info["final_info"]["episode"]
                total_success += ep_info["success_once"][mask].sum().item()
                total_eps += n_done

    env.close()
    return total_success / max(total_eps, 1)


def pad_obs(obs, target_dim):
    if obs.shape[-1] < target_dim:
        return F.pad(obs, (0, target_dim - obs.shape[-1]))
    return obs


def train(agent_type, tasks, steps_per_task, n_envs=64, device='cuda', seed=42,
          replay_ratio=0.2, grad_scale_beta=5.0, ewc_lambda=10000.0,
          mas_lambda=10000.0):
    torch.manual_seed(seed)
    np.random.seed(seed)

    # Find max obs dim across all tasks
    max_obs_dim = 0
    task_obs_dims = {}
    for t in tasks:
        print(f"  Probing {t}...", flush=True)
        tmp = gym.make(t, num_envs=1, sim_backend='gpu', render_mode=None,
                       obs_mode='state', control_mode='pd_joint_delta_pos')
        tmp_obs, _ = tmp.reset()
        task_obs_dims[t] = tmp_obs.shape[-1]
        max_obs_dim = max(max_obs_dim, tmp_obs.shape[-1])
        act_dim = tmp.action_space.shape[-1]
        tmp.close()

    obs_dim = max_obs_dim
    print(f"  obs_dim={obs_dim} (max, padded), act_dim={act_dim}")
    print(f"  Per-task obs dims: {task_obs_dims}")

    use_compression = agent_type in ('compression', 'compression_replay')
    use_replay = agent_type in ('replay', 'compression_replay', 'ewc_replay', 'mas_replay')
    use_ewc = agent_type in ('ewc', 'ewc_replay')
    use_mas = agent_type in ('mas', 'mas_replay')

    training_freq = max(n_envs, 64)
    agent = GPUSACAgent(obs_dim, act_dim, hidden=256,
                        use_compression=use_compression,
                        gamma_comp=0.01, replacement_rate=0.001,
                        grad_scale_beta=grad_scale_beta,
                        use_replay=use_replay, replay_ratio=replay_ratio,
                        use_ewc=use_ewc, ewc_lambda=ewc_lambda,
                        use_mas=use_mas, mas_lambda=mas_lambda,
                        gamma=0.8, tau=0.01, batch_size=1024,
                        training_freq=training_freq, utd=0.5,
                        device=device)
    buffer = GPUReplayBuffer(1_000_000, obs_dim, act_dim, device)

    all_results = []
    t_global = time.time()

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}: {task_name} ({steps_per_task:,} steps, {n_envs} GPU envs)")
        print(f"{'='*60}")

        env = make_env(task_name, n_envs, ignore_terminations=True)
        obs, _ = env.reset()
        obs = pad_obs(obs, obs_dim)

        total_steps = 0
        t0 = time.time()
        learning_started = task_idx > 0

        while total_steps < steps_per_task:
            if not learning_started and total_steps < 4000:
                action = 2 * torch.rand(n_envs, act_dim, device=device) - 1
            else:
                action = agent.act(obs)

            next_obs, reward, term, trunc, info = env.step(action)
            next_obs_padded = pad_obs(next_obs, obs_dim)

            real_next_obs = next_obs_padded.clone()
            stop_bootstrap = torch.zeros(n_envs, 1, device=device)
            need_final = term | trunc
            if "final_observation" in info and need_final.any():
                fo = info["final_observation"]
                real_next_obs[need_final] = pad_obs(fo[need_final], obs_dim)

            buffer.add(obs, action, reward, real_next_obs, stop_bootstrap)
            obs = next_obs_padded
            total_steps += n_envs

            if total_steps % training_freq < n_envs and buffer.size >= agent.batch_size:
                if total_steps >= 4000 or task_idx > 0:
                    learning_started = True
                    agent.update(buffer)

            if total_steps % 100_000 < n_envs:
                elapsed = time.time() - t0
                sps = total_steps / max(elapsed, 1)
                s = evaluate(agent, task_name, obs_dim, device=device)

                prev = ""
                for pi in range(task_idx):
                    ps = evaluate(agent, tasks[pi], obs_dim, device=device)
                    prev += f" | {tasks[pi][:8]}={ps:.2f}"

                rep = agent.total_replaced if use_compression else 0
                rp = f" replay={agent.replay_store.n_tasks}tasks" if use_replay else ""
                print(f"  {total_steps//1000}K ({sps:,.0f} sps): "
                      f"success={s:.2f} alpha={agent.alpha:.3f} rep={rep}{rp}{prev}")

                all_results.append({
                    'task': task_idx, 'task_name': task_name,
                    'steps': total_steps, 'success': s,
                })

        env.close()
        agent.snapshot_importance()
        agent.save_task_replay(buffer, n_samples=50_000)
        agent.compute_ewc_fisher(buffer)
        agent.compute_mas_omega(buffer)

    # Final eval
    print(f"\n{'='*60}")
    print(f"FINAL ({time.time()-t_global:.0f}s total)")
    print(f"{'='*60}")
    final = {}
    for t in tasks:
        s = evaluate(agent, t, obs_dim, device=device)
        final[t] = {'success': s}
        print(f"  {t}: success={s:.2f}")
    avg_s = np.mean([v['success'] for v in final.values()])
    print(f"  Average success: {avg_s:.2f}")

    return all_results, final


TASK_PRESETS = {
    'full5': ['PickCube-v1', 'StackCube-v1', 'PegInsertionSide-v1',
              'PushCube-v1', 'PlugCharger-v1'],
    'easy2': ['PickCube-v1', 'PushCube-v1'],
    'easy4': ['PickCube-v1', 'PushCube-v1', 'PickCube-v1', 'PushCube-v1'],  # cyclic
    'solo_peg': ['PegInsertionSide-v1'],
    'solo_stack': ['StackCube-v1'],
    'solo_plug': ['PlugCharger-v1'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', required=True,
                        choices=['standard', 'replay', 'compression', 'compression_replay',
                                 'ewc', 'ewc_replay', 'mas', 'mas_replay'])
    parser.add_argument('--steps_per_task', type=int, default=500_000)
    parser.add_argument('--n_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--tasks', default='full5',
                        help='Task preset or comma-separated list')
    parser.add_argument('--tag', default='', help='Suffix for checkpoint filename')
    parser.add_argument('--replay_ratio', type=float, default=0.2)
    parser.add_argument('--grad_scale_beta', type=float, default=5.0)
    parser.add_argument('--ewc_lambda', type=float, default=10000.0,
                        help='EWC lambda (Continual World uses 1e4 for SAC)')
    parser.add_argument('--mas_lambda', type=float, default=10000.0,
                        help='MAS lambda (Continual World uses 1e4 for SAC)')
    args = parser.parse_args()

    if args.tasks in TASK_PRESETS:
        tasks = TASK_PRESETS[args.tasks]
    else:
        tasks = args.tasks.split(',')

    print(f"GPU Continual RL: {args.agent}, {len(tasks)} tasks, "
          f"{args.steps_per_task:,} steps/task, {args.n_envs} GPU envs, seed={args.seed}")
    print(f"Tasks: {tasks}")
    print(f"replay_ratio={args.replay_ratio}, grad_scale_beta={args.grad_scale_beta}")

    results, final = train(args.agent, tasks, args.steps_per_task,
                           n_envs=args.n_envs, seed=args.seed,
                           replay_ratio=args.replay_ratio,
                           grad_scale_beta=args.grad_scale_beta,
                           ewc_lambda=args.ewc_lambda,
                           mas_lambda=args.mas_lambda)

    os.makedirs('checkpoints', exist_ok=True)
    hp_tag = ''
    if args.replay_ratio != 0.2 or args.grad_scale_beta != 5.0:
        hp_tag = f'_rr{args.replay_ratio}_b{args.grad_scale_beta}'
    if args.ewc_lambda != 10000.0 and args.agent in ('ewc', 'ewc_replay'):
        hp_tag = hp_tag + f'_l{args.ewc_lambda}'
    fname = f'checkpoints/gpu_rl_{args.agent}_{args.tasks}_s{args.seed}{hp_tag}{args.tag}.pt'
    torch.save({'results': results, 'final': final, 'config': vars(args)}, fname)
    print(f"Saved: {fname}")


if __name__ == '__main__':
    main()
