"""Train SAC on the GPU hammer env. Everything stays on GPU for max throughput."""

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

from rl.cw_tasks import make_env as make_cw_env
from rl.cw_gpu_env import OBS_DIM, ACT_DIM, MAX_EP_LEN

LOG_STD_MIN, LOG_STD_MAX = -5.0, 2.0


class ReplayBuffer:
    """GPU replay buffer. All storage is torch tensors on device."""

    def __init__(self, capacity: int, obs_dim: int, act_dim: int, device='cuda:0'):
        self.capacity = capacity
        self.device = device
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.act = torch.zeros(capacity, act_dim, device=device)
        self.rew = torch.zeros(capacity, 1, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.done = torch.zeros(capacity, 1, device=device)
        self.pos = 0
        self.size = 0

    def add_batch(self, obs, act, rew, next_obs, done):
        """Add a batch of transitions (already on GPU)."""
        n = obs.shape[0]
        if self.pos + n <= self.capacity:
            idx = slice(self.pos, self.pos + n)
        else:
            # Wrap around
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


class Actor(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
            nn.Linear(hidden, hidden), nn.LayerNorm(hidden), nn.LeakyReLU(),
        )
        self.mean = nn.Linear(hidden, act_dim)
        self.log_std = nn.Linear(hidden, act_dim)

    def forward(self, obs):
        h = self.net(obs)
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
        log_prob = (dist.log_prob(x) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob

    @torch.no_grad()
    def act(self, obs):
        action, _ = self.sample(obs)
        return action

    @torch.no_grad()
    def act_deterministic(self, obs):
        mu, _ = self.forward(obs)
        return torch.tanh(mu)


class Critic(nn.Module):
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256):
        super().__init__()
        d = obs_dim + act_dim
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


class SACAgent:
    def __init__(self, obs_dim=OBS_DIM, act_dim=ACT_DIM, hidden=256,
                 lr=3e-4, gamma=0.99, tau=0.005, batch_size=256,
                 device='cuda:0'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size

        self.actor = Actor(obs_dim, act_dim, hidden).to(device)
        self.critic = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target = Critic(obs_dim, act_dim, hidden).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        self.actor_opt = torch.optim.Adam(self.actor.parameters(), lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        self.target_entropy = -float(act_dim)
        self.log_alpha = torch.zeros(1, device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    def update(self, buffer):
        s, a, r, ns, d = buffer.sample(self.batch_size)
        alpha = self.log_alpha.exp()

        # Critic
        with torch.no_grad():
            na, nlp = self.actor.sample(ns)
            q1t, q2t = self.critic_target(ns, na)
            qt = r + (1 - d) * self.gamma * (torch.min(q1t, q2t) - alpha * nlp)

        q1, q2 = self.critic(s, a)
        cl = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
        self.critic_opt.zero_grad()
        cl.backward()
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=10.0)
        self.critic_opt.step()

        # Actor
        na2, lp2 = self.actor.sample(s)
        q1n, q2n = self.critic(s, na2)
        al = (alpha.detach() * lp2 - torch.min(q1n, q2n)).mean()
        self.actor_opt.zero_grad()
        al.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()

        # Alpha
        alpha_loss = -(self.log_alpha.exp() * (lp2 + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # Target soft update
        with torch.no_grad():
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.lerp_(p.data, self.tau)

        return {'critic_loss': cl.item(), 'actor_loss': al.item(), 'alpha': alpha.item()}


def train(n_envs=1024, total_steps=1_000_000, buffer_size=500_000,
          start_steps=10_000, update_freq=1, seed=42):
    """Train SAC on GPU hammer. n_envs=1024 means each env.step gives 1024 transitions,
    which enables high UTD on GPU without bottlenecking on environment steps."""
    torch.manual_seed(seed)
    np.random.seed(seed)

    env = make_cw_env('hammer', n_envs=n_envs)
    obs = env.reset()

    agent = SACAgent(batch_size=1024)  # Large batch to match GPU throughput
    buffer = ReplayBuffer(buffer_size, OBS_DIM, ACT_DIM)

    t_start = time.time()
    total_env_steps = 0
    total_grad_steps = 0
    last_log = time.time()
    last_log_step = 0

    while total_env_steps < total_steps:
        # Collect one batch of transitions
        if total_env_steps < start_steps:
            action = torch.rand(n_envs, ACT_DIM, device='cuda:0') * 2 - 1
        else:
            action = agent.actor.act(obs)

        next_obs, reward, done, info = env.auto_reset_step(action)
        real_next_obs = info['real_next_obs']

        # Add to buffer
        buffer.add_batch(obs, action, reward, real_next_obs, done)
        obs = next_obs
        total_env_steps += n_envs

        # Train
        if total_env_steps >= start_steps and buffer.size >= agent.batch_size:
            # UTD = gradient updates / env steps
            # With n_envs transitions per env.step, we do (n_envs * update_freq)
            # gradient updates per env.step call.
            # The CW protocol uses UTD=1 (1 grad step per env step).
            # At n_envs=1024, update_freq=1 means 1024 grad updates per 1024 env steps.
            n_updates = max(1, int(n_envs * update_freq))
            for _ in range(n_updates):
                agent.update(buffer)
                total_grad_steps += 1

        # Log periodically
        if total_env_steps % (n_envs * 50) < n_envs:
            elapsed = time.time() - t_start
            sps = total_env_steps / max(elapsed, 1e-6)
            recent_dt = time.time() - last_log
            recent_sps = (total_env_steps - last_log_step) / max(recent_dt, 1e-6)
            last_log = time.time()
            last_log_step = total_env_steps

            succ_once = info['success_once'].float().mean().item()
            ep_ret = info['ep_return'].mean().item()
            print(f'[{total_env_steps//1000:5d}K] '
                  f'sps={sps:,.0f} (recent={recent_sps:,.0f}) '
                  f'succ={succ_once:.3f} ep_ret={ep_ret:.2f} '
                  f'buf={buffer.size} upd={total_grad_steps} '
                  f'alpha={agent.alpha:.3f}', flush=True)

    print(f'\nTotal time: {time.time()-t_start:.0f}s')
    print(f'Final sps: {total_steps / (time.time()-t_start):,.0f}')

    # Final eval
    eval_env = make_cw_env('hammer', n_envs=32)
    eval_obs = eval_env.reset()
    successes = torch.zeros(32, device='cuda:0')
    for _ in range(MAX_EP_LEN):
        a = agent.actor.act_deterministic(eval_obs)
        eval_obs, _, done, info = eval_env.auto_reset_step(a)
        successes = torch.max(successes, info['success_once'].float())
    print(f'Eval success rate (32 envs): {successes.mean().item():.2f}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n_envs', type=int, default=1024)
    parser.add_argument('--total_steps', type=int, default=500_000)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--update_freq', type=float, default=1.0,
                        help='gradient updates per env step')
    args = parser.parse_args()

    train(n_envs=args.n_envs, total_steps=args.total_steps,
          seed=args.seed, update_freq=args.update_freq)


if __name__ == '__main__':
    main()
