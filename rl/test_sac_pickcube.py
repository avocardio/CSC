"""Minimal SAC test on PickCube — matches ManiSkill3 reference implementation."""

import os
os.environ['VK_ICD_FILENAMES'] = '/usr/share/vulkan/icd.d/lvp_icd.x86_64.json'
os.environ['LD_LIBRARY_PATH'] = '/usr/lib/wsl/lib:' + os.environ.get('LD_LIBRARY_PATH', '')

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

print("Starting...", flush=True)

device = 'cuda'
n_envs = 32
training_freq = 64
utd_ratio = 0.5
grad_steps = int(training_freq * utd_ratio)  # 32

print(f"Creating env with {n_envs} envs...", flush=True)
raw_env = gym.make('PickCube-v1', num_envs=n_envs, sim_backend='gpu',
                    render_mode=None, obs_mode='state',
                    control_mode='pd_joint_delta_pos')
env = ManiSkillVectorEnv(raw_env, n_envs, ignore_terminations=True,
                         record_metrics=True)
obs, _ = env.reset()
obs_dim = obs.shape[-1]
act_dim = env.single_action_space.shape[-1]
print(f'obs_dim={obs_dim}, act_dim={act_dim}', flush=True)

LOG_STD_MIN, LOG_STD_MAX = -5, 2

class Actor(nn.Module):
    def __init__(self):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU(),
            nn.Linear(256, 256), nn.ReLU())
        self.mean = nn.Linear(256, act_dim)
        self.log_std = nn.Linear(256, act_dim)

    def forward(self, x):
        h = self.net(x)
        mu = self.mean(h)
        ls = torch.tanh(self.log_std(h))
        ls = LOG_STD_MIN + 0.5 * (LOG_STD_MAX - LOG_STD_MIN) * (ls + 1)
        return mu, ls

    def sample(self, x):
        mu, ls = self.forward(x)
        dist = Normal(mu, ls.exp())
        a = dist.rsample()
        action = torch.tanh(a)
        lp = (dist.log_prob(a) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, lp

    def get_eval_action(self, x):
        mu, _ = self.forward(x)
        return torch.tanh(mu)

class Critic(nn.Module):
    def __init__(self):
        super().__init__()
        d = obs_dim + act_dim
        self.q1 = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))
        self.q2 = nn.Sequential(nn.Linear(d, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 256), nn.ReLU(), nn.Linear(256, 1))

    def forward(self, o, a):
        x = torch.cat([o, a], -1)
        return self.q1(x), self.q2(x)

print("Creating networks...", flush=True)
actor = Actor().to(device)
critic = Critic().to(device)
critic_target = Critic().to(device)
critic_target.load_state_dict(critic.state_dict())

actor_opt = torch.optim.Adam(actor.parameters(), lr=3e-4)
critic_opt = torch.optim.Adam(critic.parameters(), lr=3e-4)
log_alpha = torch.zeros(1, device=device, requires_grad=True)
alpha_opt = torch.optim.Adam([log_alpha], lr=3e-4)
target_entropy = -float(act_dim)

buf_cap = 1_000_000
buf_o = torch.zeros(buf_cap, obs_dim, device=device)
buf_a = torch.zeros(buf_cap, act_dim, device=device)
buf_r = torch.zeros(buf_cap, 1, device=device)
buf_no = torch.zeros(buf_cap, obs_dim, device=device)
buf_d = torch.zeros(buf_cap, 1, device=device)  # stop_bootstrap flag
buf_pos, buf_len = 0, 0

gamma, tau, bs = 0.8, 0.01, 1024
total = 500_000
start_steps = 4000

print("Starting training...", flush=True)
t0 = time.time()
step = 0
learning_started = False

while step < total:
    # Act
    if not learning_started:
        action = 2 * torch.rand(n_envs, act_dim, device=device) - 1
    else:
        with torch.no_grad():
            action, _ = actor.sample(obs)

    next_obs, reward, term, trunc, info = env.step(action)

    # Handle final observations for auto-reset envs (bootstrap_at_done='always')
    real_next_obs = next_obs.clone()
    stop_bootstrap = torch.zeros(n_envs, 1, device=device)  # always bootstrap
    need_final = term | trunc
    if "final_observation" in info:
        real_next_obs[need_final] = info["final_observation"][need_final]

    # Store in buffer
    n = n_envs
    idx = torch.arange(buf_pos, buf_pos + n) % buf_cap
    buf_o[idx] = obs
    buf_a[idx] = action
    buf_r[idx] = reward.unsqueeze(-1) if reward.dim() == 1 else reward
    buf_no[idx] = real_next_obs  # true next obs, not post-reset
    buf_d[idx] = stop_bootstrap  # always 0 = always bootstrap
    buf_pos = (buf_pos + n) % buf_cap
    buf_len = min(buf_len + n, buf_cap)
    obs = next_obs  # use post-reset obs for next step
    step += n

    # Train every training_freq transitions
    if step % training_freq < n_envs and buf_len >= bs and step >= start_steps:
        learning_started = True
        for _ in range(grad_steps):
            idx = torch.randint(0, buf_len, (bs,), device=device)
            s, a, r, ns, d = buf_o[idx], buf_a[idx], buf_r[idx], buf_no[idx], buf_d[idx]

            with torch.no_grad():
                na, nlp = actor.sample(ns)
                q1t, q2t = critic_target(ns, na)
                qt = r + (1 - d) * gamma * (torch.min(q1t, q2t) - log_alpha.exp() * nlp)

            q1, q2 = critic(s, a)
            cl = F.mse_loss(q1, qt) + F.mse_loss(q2, qt)
            critic_opt.zero_grad(); cl.backward(); critic_opt.step()

            na2, lp2 = actor.sample(s)
            q1n, q2n = critic(s, na2)
            al = (log_alpha.exp() * lp2 - torch.min(q1n, q2n)).mean()
            actor_opt.zero_grad(); al.backward(); actor_opt.step()

            aloss = -(log_alpha.exp() * (lp2 + target_entropy).detach()).mean()
            alpha_opt.zero_grad(); aloss.backward(); alpha_opt.step()

            for p, tp in zip(critic.parameters(), critic_target.parameters()):
                tp.data.lerp_(p.data, tau)

    if step % 50000 < n_envs:
        sps = step / max(time.time() - t0, 1)
        # Get success from episode metrics
        succ = info.get('success', torch.zeros(n_envs))
        if isinstance(succ, torch.Tensor):
            s = succ.float().mean().item()
        else:
            s = 0
        alpha = log_alpha.exp().item()
        print(f'{step//1000}K ({sps:.0f} sps): success={s:.2f}, alpha={alpha:.3f}', flush=True)

env.close()
print(f'Done in {time.time()-t0:.0f}s', flush=True)
