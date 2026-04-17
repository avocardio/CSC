"""SAC with self-compression for continual RL.

The compression objective identifies underutilized units in the policy/critic
networks. These units are replaced with fresh random weights, maintaining
plasticity. This is the RL version of our approach.

Key difference from supervised CL:
- In RL, data distribution is non-stationary WITHIN each task
- The network must continuously reorganize representations
- Compression naturally identifies which features are currently active vs dormant
- This is a continuous plasticity maintenance mechanism, not just inter-task protection
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
from rl.sac import ReplayBuffer


class CompressedMLP(nn.Module):
    """MLP with per-unit importance (bit-depth) parameters."""

    def __init__(self, input_dim, hidden_dim, output_dim, n_layers=2):
        super().__init__()
        self.layers = nn.ModuleList()
        self.importance = nn.ParameterList()

        # Build layers
        dims = [input_dim] + [hidden_dim] * n_layers + [output_dim]
        for i in range(len(dims) - 1):
            self.layers.append(nn.Linear(dims[i], dims[i+1]))
            if i < len(dims) - 2:  # importance for hidden layers only
                self.importance.append(
                    nn.Parameter(torch.full((dims[i+1],), 8.0)))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i < len(self.layers) - 1:
                x = F.relu(x)
        return x

    def compute_compression_loss(self):
        total = sum(b.clamp(min=0).sum() for b in self.importance)
        count = sum(b.numel() for b in self.importance)
        return total / max(count, 1)

    def get_unit_importance(self, layer_idx):
        """Get importance values for a hidden layer."""
        if layer_idx < len(self.importance):
            return self.importance[layer_idx].clamp(min=0).detach()
        return None


class CompressedGaussianPolicy(nn.Module):
    """Gaussian policy with compression-based importance."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.trunk = CompressedMLP(obs_dim, hidden_dim, hidden_dim, n_layers=2)
        # Override: trunk outputs hidden_dim, not hidden_dim again
        # Simpler: just use standard layers with importance params
        self.fc1 = nn.Linear(obs_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.mean_head = nn.Linear(hidden_dim, act_dim)
        self.log_std_head = nn.Linear(hidden_dim, act_dim)

        self.importance = nn.ParameterList([
            nn.Parameter(torch.full((hidden_dim,), 8.0)),
            nn.Parameter(torch.full((hidden_dim,), 8.0)),
        ])

    def forward(self, obs):
        h = F.relu(self.fc1(obs))
        h = F.relu(self.fc2(h))
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

    def compression_loss(self):
        total = sum(b.clamp(min=0).sum() for b in self.importance)
        count = sum(b.numel() for b in self.importance)
        return total / max(count, 1)


class CompressedTwinQ(nn.Module):
    """Twin Q-networks with compression."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        input_dim = obs_dim + act_dim
        self.q1_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q1_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q1_out = nn.Linear(hidden_dim, 1)

        self.q2_fc1 = nn.Linear(input_dim, hidden_dim)
        self.q2_fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.q2_out = nn.Linear(hidden_dim, 1)

        self.importance = nn.ParameterList([
            nn.Parameter(torch.full((hidden_dim,), 8.0)),  # q1 layer 1
            nn.Parameter(torch.full((hidden_dim,), 8.0)),  # q1 layer 2
            nn.Parameter(torch.full((hidden_dim,), 8.0)),  # q2 layer 1
            nn.Parameter(torch.full((hidden_dim,), 8.0)),  # q2 layer 2
        ])

    def forward(self, obs, action):
        x = torch.cat([obs, action], dim=-1)
        q1 = F.relu(self.q1_fc1(x))
        q1 = F.relu(self.q1_fc2(q1))
        q1 = self.q1_out(q1)

        q2 = F.relu(self.q2_fc1(x))
        q2 = F.relu(self.q2_fc2(q2))
        q2 = self.q2_out(q2)
        return q1, q2

    def compression_loss(self):
        total = sum(b.clamp(min=0).sum() for b in self.importance)
        count = sum(b.numel() for b in self.importance)
        return total / max(count, 1)


class CompressionSAC:
    """SAC agent with self-compression for plasticity maintenance."""

    def __init__(self, obs_dim, act_dim, hidden_dim=256, lr=3e-4,
                 gamma_rl=0.99, tau=0.005, alpha_init=0.2, auto_alpha=True,
                 gamma_comp=0.01, replacement_rate=0.001, maturity_threshold=1000,
                 device='cuda'):
        self.device = device
        self.gamma_rl = gamma_rl
        self.tau = tau
        self.gamma_comp = gamma_comp
        self.replacement_rate = replacement_rate
        self.maturity_threshold = maturity_threshold

        # Networks with compression
        self.policy = CompressedGaussianPolicy(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = CompressedTwinQ(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = CompressedTwinQ(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Separate optimizers for network params and importance params
        policy_params = [p for n, p in self.policy.named_parameters() if 'importance' not in n]
        policy_imp = list(self.policy.importance)
        critic_params = [p for n, p in self.critic.named_parameters() if 'importance' not in n]
        critic_imp = list(self.critic.importance)

        self.policy_optimizer = torch.optim.Adam(policy_params, lr=lr)
        self.critic_optimizer = torch.optim.Adam(critic_params, lr=lr)
        self.importance_optimizer = torch.optim.Adam(
            list(policy_imp) + list(critic_imp), lr=0.01)

        # Entropy tuning
        self.auto_alpha = auto_alpha
        if auto_alpha:
            self.target_entropy = -act_dim
            self.log_alpha = torch.tensor(np.log(alpha_init), device=device, requires_grad=True)
            self.alpha_optimizer = torch.optim.Adam([self.log_alpha], lr=lr)
            self.alpha = self.log_alpha.exp().item()
        else:
            self.alpha = alpha_init

        # Unit utility tracking (EMA of activation magnitude)
        self.utility = {}
        self.ages = {}
        self._init_utility_tracking()
        self.total_replaced = 0
        self.update_count = 0

    def _init_utility_tracking(self):
        """Initialize per-unit utility and age tracking."""
        for name in ['policy_fc1', 'policy_fc2']:
            dim = self.policy.fc1.out_features
            self.utility[name] = torch.zeros(dim, device=self.device)
            self.ages[name] = torch.zeros(dim, device=self.device)

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
        self.update_count += 1

        states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        # --- Critic update ---
        with torch.no_grad():
            next_actions, next_log_probs, _ = self.policy.sample(next_states)
            q1_next, q2_next = self.critic_target(next_states, next_actions)
            q_next = torch.min(q1_next, q2_next) - self.alpha * next_log_probs
            q_target = rewards + (1 - dones) * self.gamma_rl * q_next

        q1, q2 = self.critic(states, actions)
        critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

        # Add compression loss to critic
        critic_comp = self.gamma_comp * self.critic.compression_loss()
        total_critic_loss = critic_loss + critic_comp

        self.critic_optimizer.zero_grad()
        self.importance_optimizer.zero_grad()
        total_critic_loss.backward()
        self.critic_optimizer.step()

        # --- Policy update ---
        new_actions, log_probs, _ = self.policy.sample(states)
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)
        policy_loss = (self.alpha * log_probs - q_new).mean()

        # Add compression loss to policy
        policy_comp = self.gamma_comp * self.policy.compression_loss()
        total_policy_loss = policy_loss + policy_comp

        self.policy_optimizer.zero_grad()
        total_policy_loss.backward()
        self.policy_optimizer.step()

        # Update importance params
        self.importance_optimizer.step()

        # --- Alpha update ---
        if self.auto_alpha:
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()
            self.alpha = self.log_alpha.exp().item()

        # --- Soft target update ---
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        # --- Unit replacement (compression-guided) ---
        if self.update_count % 100 == 0:  # every 100 updates
            self._replace_low_importance_units()

        return {
            'critic_loss': critic_loss.item(),
            'policy_loss': policy_loss.item(),
            'alpha': self.alpha,
            'policy_comp': policy_comp.item(),
        }

    def _replace_low_importance_units(self):
        """Replace units with lowest learned bit-depth (importance)."""
        replaced = 0
        for layer_idx, (layer, imp) in enumerate([
            (self.policy.fc1, self.policy.importance[0]),
            (self.policy.fc2, self.policy.importance[1]),
        ]):
            bits = imp.clamp(min=0).detach()
            n_units = bits.shape[0]

            # How many to replace
            n_replace = self.replacement_rate * n_units
            if n_replace < 1:
                if torch.rand(1).item() < n_replace:
                    n_replace = 1
                else:
                    continue
            n_replace = int(n_replace)

            # Find lowest importance units
            _, lowest = torch.topk(-bits, n_replace)

            with torch.no_grad():
                # Reinitialize input weights
                nn.init.kaiming_normal_(layer.weight.data[lowest])
                layer.bias.data[lowest] = 0
                # Reset bit-depth to init value
                imp.data[lowest] = 8.0

            replaced += n_replace

        self.total_replaced += replaced
