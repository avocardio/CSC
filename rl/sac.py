"""Soft Actor-Critic (SAC) implementation for Meta-World tasks.

Clean, minimal SAC with:
- Twin Q-networks
- Automatic entropy tuning
- Optional self-compression on the policy/critic networks
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from collections import deque
import random


class ReplayBuffer:
    """Simple replay buffer for SAC."""

    def __init__(self, capacity=1_000_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            torch.FloatTensor(np.array(states)),
            torch.FloatTensor(np.array(actions)),
            torch.FloatTensor(np.array(rewards)).unsqueeze(1),
            torch.FloatTensor(np.array(next_states)),
            torch.FloatTensor(np.array(dones)).unsqueeze(1),
        )

    def __len__(self):
        return len(self.buffer)


class MLP(nn.Module):
    """Simple MLP for SAC policy and critic."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        layers = [nn.Linear(input_dim, hidden_dim), nn.ReLU()]
        for _ in range(n_layers - 1):
            layers.extend([nn.Linear(hidden_dim, hidden_dim), nn.ReLU()])
        layers.append(nn.Linear(hidden_dim, output_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


class GaussianPolicy(nn.Module):
    """Gaussian policy for SAC."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

    def forward(self, obs):
        h = self.trunk(obs)
        mean = self.mean_head(h)
        log_std = self.log_std_head(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)
        return mean, log_std

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)
        log_prob = log_prob.sum(-1, keepdim=True)
        return action, log_prob, mean


class TwinQNetwork(nn.Module):
    """Twin Q-networks for SAC."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = MLP(obs_dim + act_dim, hidden_dim, 1)
        self.q2 = MLP(obs_dim + act_dim, hidden_dim, 1)

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        return self.q1(x), self.q2(x)


class SAC:
    """Soft Actor-Critic agent."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, lr=3e-4,
                 gamma=0.99, tau=0.005, alpha_init=0.2, auto_alpha=True,
                 device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau

        # Networks
        self.policy = GaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = TwinQNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = TwinQNetwork(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        self.policy_optimizer = torch.optim.Adam(self.policy.parameters(), lr=lr)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=lr)

        # Automatic entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -act_dim
            self.log_alpha = torch.tensor(np.log(alpha_init), device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

    def select_action(self, obs, evaluate=False):
        with torch.no_grad():
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(self.device)
            if evaluate:
                mean, _ = self.policy(obs_t)
                return torch.tanh(mean).cpu().numpy()[0]
            else:
                action, _, _ = self.policy.sample(obs_t)
                return action.cpu().numpy()[0]

    def update(self, replay_buffer, batch_size=256):
        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # Update critic
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update policy
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        self.policy_optimizer.zero_grad()
        policy_loss.backward()
        self.policy_optimizer.step()

        # Update alpha
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
        }

    def save(self, path):
        torch.save({
            'policy': self.policy.state_dict(),
            'critic': self.critic.state_dict(),
            'critic_target': self.critic_target.state_dict(),
        }, path)

    def load(self, path):
        data = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(data['policy'])
        self.critic.load_state_dict(data['critic'])
        self.critic_target.load_state_dict(data['critic_target'])
