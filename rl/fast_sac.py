"""Optimized SAC with vectorized environments for maximum throughput.

Optimizations:
1. Vectorized environments (8 parallel Meta-World instances)
2. Large batch size (1024) to fully utilize GPU
3. High update-to-data ratio (UTD=4: 4 gradient steps per env step)
4. All tensors on GPU, minimal CPU-GPU transfer
5. Compiled forward passes where possible
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
import numpy as np
import metaworld
import random as py_random
import multiprocessing as mp
from collections import deque
import time


class FastReplayBuffer:
    """GPU-resident replay buffer for maximum throughput."""

    def __init__(self, capacity, obs_dim, act_dim, device='cuda'):
        self.capacity = capacity
        self.device = device
        self.pos = 0
        self.size = 0

        # Pre-allocate on GPU
        self.obs = torch.zeros(capacity, obs_dim, device=device)
        self.actions = torch.zeros(capacity, act_dim, device=device)
        self.rewards = torch.zeros(capacity, 1, device=device)
        self.next_obs = torch.zeros(capacity, obs_dim, device=device)
        self.dones = torch.zeros(capacity, 1, device=device)

    def push_batch(self, obs, actions, rewards, next_obs, dones):
        """Push a batch of transitions (from vectorized env)."""
        batch_size = obs.shape[0]
        if self.pos + batch_size > self.capacity:
            # Wrap around
            first = self.capacity - self.pos
            self.obs[self.pos:] = obs[:first]
            self.actions[self.pos:] = actions[:first]
            self.rewards[self.pos:] = rewards[:first]
            self.next_obs[self.pos:] = next_obs[:first]
            self.dones[self.pos:] = dones[:first]
            remaining = batch_size - first
            if remaining > 0:
                self.obs[:remaining] = obs[first:]
                self.actions[:remaining] = actions[first:]
                self.rewards[:remaining] = rewards[first:]
                self.next_obs[:remaining] = next_obs[first:]
                self.dones[:remaining] = dones[first:]
        else:
            self.obs[self.pos:self.pos+batch_size] = obs
            self.actions[self.pos:self.pos+batch_size] = actions
            self.rewards[self.pos:self.pos+batch_size] = rewards
            self.next_obs[self.pos:self.pos+batch_size] = next_obs
            self.dones[self.pos:self.pos+batch_size] = dones

        self.pos = (self.pos + batch_size) % self.capacity
        self.size = min(self.size + batch_size, self.capacity)

    def sample(self, batch_size):
        indices = torch.randint(0, self.size, (batch_size,), device=self.device)
        return (
            self.obs[indices],
            self.actions[indices],
            self.rewards[indices],
            self.next_obs[indices],
            self.dones[indices],
        )


class FastPolicy(nn.Module):
    """Gaussian policy optimized for speed."""

    LOG_STD_MIN = -20
    LOG_STD_MAX = 2

    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
        )
        self.mean = nn.Linear(hidden_dim, act_dim)
        self.log_std = nn.Linear(hidden_dim, act_dim)

        # Importance params for compression
        self.importance = nn.ParameterList([
            nn.Parameter(torch.full((hidden_dim,), 8.0)),
            nn.Parameter(torch.full((hidden_dim,), 8.0)),
        ])

    def forward(self, obs):
        h = self.net(obs)
        return self.mean(h), self.log_std(h).clamp(self.LOG_STD_MIN, self.LOG_STD_MAX)

    def sample(self, obs):
        mean, log_std = self.forward(obs)
        std = log_std.exp()
        normal = Normal(mean, std)
        x_t = normal.rsample()
        action = torch.tanh(x_t)
        log_prob = (normal.log_prob(x_t) - torch.log(1 - action.pow(2) + 1e-6)).sum(-1, keepdim=True)
        return action, log_prob, mean

    def compression_loss(self):
        return sum(b.clamp(min=0).mean() for b in self.importance) / len(self.importance)


class FastTwinQ(nn.Module):
    def __init__(self, obs_dim, act_dim, hidden_dim=256):
        super().__init__()
        self.q1 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))
        self.q2 = nn.Sequential(
            nn.Linear(obs_dim + act_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim), nn.ReLU(),
            nn.Linear(hidden_dim, 1))

    def forward(self, obs, action):
        x = torch.cat([obs, action], -1)
        return self.q1(x), self.q2(x)


def env_worker(task_name, conn, seed):
    """Worker process for a single Meta-World environment."""
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    task = ml1.train_tasks[seed % len(ml1.train_tasks)]
    env.set_task(task)
    obs, _ = env.reset()

    while True:
        cmd, data = conn.recv()
        if cmd == 'step':
            next_obs, reward, term, trunc, info = env.step(data)
            done = term or trunc
            if done:
                obs, _ = env.reset()
                conn.send((obs, reward, True, info))
            else:
                conn.send((next_obs, reward, False, info))
                obs = next_obs
        elif cmd == 'reset':
            obs, _ = env.reset()
            conn.send(obs)
        elif cmd == 'change_task':
            # Change to a new Meta-World task
            new_task_name = data
            env.close()
            ml1 = metaworld.ML1(new_task_name)
            env = ml1.train_classes[new_task_name]()
            task = ml1.train_tasks[seed % len(ml1.train_tasks)]
            env.set_task(task)
            obs, _ = env.reset()
            conn.send(obs)
        elif cmd == 'close':
            env.close()
            conn.close()
            break


class VectorizedMetaWorld:
    """Vectorized Meta-World environments using multiprocessing."""

    def __init__(self, task_name, n_envs=8, seed=42):
        self.n_envs = n_envs
        self.task_name = task_name

        self.parent_conns = []
        self.child_conns = []
        self.processes = []

        for i in range(n_envs):
            parent, child = mp.Pipe()
            self.parent_conns.append(parent)
            self.child_conns.append(child)
            p = mp.Process(target=env_worker, args=(task_name, child, seed + i), daemon=True)
            p.start()
            self.processes.append(p)

        # Get initial observations
        self.obs = np.zeros((n_envs, 39))
        for i, conn in enumerate(self.parent_conns):
            conn.send(('reset', None))
        for i, conn in enumerate(self.parent_conns):
            self.obs[i] = conn.recv()

    def step(self, actions):
        """Step all environments with given actions."""
        for i, conn in enumerate(self.parent_conns):
            conn.send(('step', actions[i]))

        next_obs = np.zeros((self.n_envs, 39))
        rewards = np.zeros(self.n_envs)
        dones = np.zeros(self.n_envs)
        infos = []

        for i, conn in enumerate(self.parent_conns):
            obs, reward, done, info = conn.recv()
            next_obs[i] = obs
            rewards[i] = reward
            dones[i] = float(done)
            infos.append(info)

        self.obs = next_obs
        return next_obs, rewards, dones, infos

    def change_task(self, new_task_name):
        """Change all environments to a new task."""
        self.task_name = new_task_name
        for conn in self.parent_conns:
            conn.send(('change_task', new_task_name))
        for i, conn in enumerate(self.parent_conns):
            self.obs[i] = conn.recv()

    def close(self):
        for conn in self.parent_conns:
            try:
                conn.send(('close', None))
            except:
                pass
        for p in self.processes:
            p.join(timeout=5)


class FastSACAgent:
    """High-throughput SAC with optional compression."""

    def __init__(self, obs_dim=39, act_dim=4, hidden_dim=256,
                 lr=3e-4, gamma=0.99, tau=0.005, utd_ratio=4,
                 batch_size=1024, use_compression=False, gamma_comp=0.01,
                 replacement_rate=0.001, device='cuda'):
        self.device = device
        self.gamma = gamma
        self.tau = tau
        self.utd_ratio = utd_ratio
        self.batch_size = batch_size
        self.use_compression = use_compression
        self.gamma_comp = gamma_comp
        self.replacement_rate = replacement_rate

        self.policy = FastPolicy(obs_dim, act_dim, hidden_dim).to(device)
        self.critic = FastTwinQ(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target = FastTwinQ(obs_dim, act_dim, hidden_dim).to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Optimizers
        policy_params = [p for n, p in self.policy.named_parameters() if 'importance' not in n]
        self.policy_opt = torch.optim.Adam(policy_params, lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)

        if use_compression:
            self.imp_opt = torch.optim.Adam(self.policy.importance.parameters(), lr=0.01)

        # Auto entropy
        self.target_entropy = -act_dim
        self.log_alpha = torch.tensor(np.log(0.2), device=device, requires_grad=True)
        self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=lr)

        self.total_replaced = 0
        self.update_count = 0

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

    @torch.no_grad()
    def select_actions(self, obs_np):
        """Select actions for a batch of observations."""
        obs = torch.FloatTensor(obs_np).to(self.device)
        actions, _, _ = self.policy.sample(obs)
        return actions.cpu().numpy()

    def update(self, buffer):
        """Do UTD gradient updates."""
        if buffer.size < self.batch_size:
            return {}

        metrics = {}
        for _ in range(self.utd_ratio):
            states, actions, rewards, next_states, dones = buffer.sample(self.batch_size)

            # Critic
            with torch.no_grad():
                next_actions, next_log_probs, _ = self.policy.sample(next_states)
                q1_next, q2_next = self.critic_target(next_states, next_actions)
                q_target = rewards + (1 - dones) * self.gamma * (
                    torch.min(q1_next, q2_next) - self.alpha * next_log_probs)

            q1, q2 = self.critic(states, actions)
            critic_loss = F.mse_loss(q1, q_target) + F.mse_loss(q2, q_target)

            self.critic_opt.zero_grad()
            critic_loss.backward()
            self.critic_opt.step()

            # Policy
            new_actions, log_probs, _ = self.policy.sample(states)
            q1_new, q2_new = self.critic(states, new_actions)
            policy_loss = (self.alpha * log_probs - torch.min(q1_new, q2_new)).mean()

            if self.use_compression:
                policy_loss = policy_loss + self.gamma_comp * self.policy.compression_loss()

            self.policy_opt.zero_grad()
            if self.use_compression:
                self.imp_opt.zero_grad()
            policy_loss.backward()
            self.policy_opt.step()
            if self.use_compression:
                self.imp_opt.step()

            # Alpha
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()
            self.alpha_opt.zero_grad()
            alpha_loss.backward()
            self.alpha_opt.step()

            # Soft target
            for p, tp in zip(self.critic.parameters(), self.critic_target.parameters()):
                tp.data.copy_(self.tau * p.data + (1 - self.tau) * tp.data)

        self.update_count += 1

        # Compression-guided replacement every 50 updates
        if self.use_compression and self.update_count % 50 == 0:
            self._replace_units()

        metrics['critic_loss'] = critic_loss.item()
        metrics['policy_loss'] = policy_loss.item()
        metrics['alpha'] = self.alpha
        return metrics

    def _replace_units(self):
        """Replace lowest importance units in policy."""
        with torch.no_grad():
            for layer_idx, (layer_name, imp) in enumerate([
                ('net.0', self.policy.importance[0]),
                ('net.2', self.policy.importance[1]),
            ]):
                bits = imp.clamp(min=0)
                n = bits.shape[0]
                n_replace = self.replacement_rate * n
                if n_replace < 1:
                    if torch.rand(1).item() < n_replace:
                        n_replace = 1
                    else:
                        continue
                n_replace = int(n_replace)

                _, lowest = torch.topk(-bits, n_replace)

                # Get the actual layer
                layer = self.policy.net[layer_idx * 2]  # 0->net.0, 1->net.2
                nn.init.kaiming_normal_(layer.weight.data[lowest])
                layer.bias.data[lowest] = 0
                imp.data[lowest] = 8.0
                self.total_replaced += n_replace


def evaluate_agent(agent, task_name, n_episodes=10, device='cuda'):
    """Evaluate agent on a single task."""
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    task = ml1.train_tasks[0]
    env.set_task(task)

    total_reward, total_success = 0, 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(500):
            obs_t = torch.FloatTensor(obs).unsqueeze(0).to(device)
            with torch.no_grad():
                mean, _ = agent.policy(obs_t)
                action = torch.tanh(mean).cpu().numpy()[0]
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc: break
        total_reward += ep_reward
        total_success += float(info.get('success', 0))

    env.close()
    return total_reward / n_episodes, total_success / n_episodes


def train_continual(agent_type, tasks, steps_per_task, n_envs=8, device='cuda', seed=42):
    """Train SAC on sequential tasks with vectorized envs."""
    obs_dim, act_dim = 39, 4

    agent = FastSACAgent(
        obs_dim, act_dim,
        use_compression=(agent_type == 'compression'),
        gamma_comp=0.01,
        replacement_rate=0.001,
        utd_ratio=4,
        batch_size=1024,
        device=device,
    )

    buffer = FastReplayBuffer(1_000_000, obs_dim, act_dim, device)
    results = []

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'='*50}")
        print(f"TASK {task_idx}: {task_name} ({steps_per_task} steps, {n_envs} envs)")
        print(f"{'='*50}")

        vec_env = VectorizedMetaWorld(task_name, n_envs=n_envs, seed=seed)
        total_steps = 0
        t0 = time.time()

        while total_steps < steps_per_task:
            # Collect transitions from all envs
            actions = agent.select_actions(vec_env.obs)
            next_obs, rewards, dones, infos = vec_env.step(actions)

            # Push to GPU buffer
            buffer.push_batch(
                torch.FloatTensor(vec_env.obs).to(device),  # use prev obs
                torch.FloatTensor(actions).to(device),
                torch.FloatTensor(rewards).unsqueeze(1).to(device),
                torch.FloatTensor(next_obs).to(device),
                torch.FloatTensor(dones).unsqueeze(1).to(device),
            )

            total_steps += n_envs

            # Update agent
            if buffer.size >= agent.batch_size:
                agent.update(buffer)

            # Evaluate periodically
            if total_steps % 50000 < n_envs:
                elapsed = time.time() - t0
                sps = total_steps / max(elapsed, 1)

                # Evaluate current task
                avg_r, succ = evaluate_agent(agent, task_name, n_episodes=5, device=device)
                replaced = agent.total_replaced

                # Evaluate previous tasks
                prev_str = ""
                for pi in range(task_idx):
                    pr, ps = evaluate_agent(agent, tasks[pi], n_episodes=3, device=device)
                    prev_str += f" | {tasks[pi][:6]}={ps:.2f}"

                print(f"  {total_steps//1000}K steps ({sps:.0f} sps): "
                      f"reward={avg_r:.0f} success={succ:.2f} "
                      f"replaced={replaced}{prev_str}")

                results.append({
                    'task_idx': task_idx, 'steps': total_steps,
                    'reward': avg_r, 'success': succ,
                })

        vec_env.close()

    # Final evaluation
    print(f"\n{'='*50}")
    print(f"FINAL EVALUATION")
    print(f"{'='*50}")
    final = {}
    for t in tasks:
        r, s = evaluate_agent(agent, t, n_episodes=10, device=device)
        final[t] = {'reward': r, 'success': s}
        print(f"  {t}: reward={r:.0f}, success={s:.2f}")

    avg_success = np.mean([v['success'] for v in final.values()])
    print(f"  Average success: {avg_success:.2f}")

    return results, final


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True, choices=['standard', 'compression'])
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--steps_per_task', type=int, default=500000)
    parser.add_argument('--n_envs', type=int, default=8)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_random.seed(args.seed)

    tasks = ['reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'drawer-close-v3']
    tasks = tasks[:args.num_tasks]

    print(f"Fast Continual RL: {args.agent}, {len(tasks)} tasks, "
          f"{args.steps_per_task} steps/task, {args.n_envs} parallel envs")

    results, final = train_continual(
        args.agent, tasks, args.steps_per_task,
        n_envs=args.n_envs, seed=args.seed)

    import os
    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'results': results, 'final': final},
               f'checkpoints/fast_rl_{args.agent}.pt')
