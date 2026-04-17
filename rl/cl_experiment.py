"""Continual Learning experiment runner on GPU CW tasks.

Runs a sequence of tasks, training SAC with various CL methods:
- finetune (no CL, baseline)
- ewc
- mas
- replay
- compression (ours)
- compression_replay (ours full)

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
    def __init__(self, method='finetune', lr=3e-4, gamma=0.99, tau=0.005,
                 batch_size=256, replay_ratio=0.25, cl_reg_coef=1e5,
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
        self.use_compression = method in ('compression', 'compression_replay')
        self.use_replay = method in ('replay', 'compression_replay')
        self.use_ewc = method == 'ewc'

        self.actor = Actor(use_compression=self.use_compression).to(device)
        self.critic = Critic().to(device)
        self.critic_target = Critic().to(device)
        self.critic_target.load_state_dict(self.critic.state_dict())
        for p in self.critic_target.parameters():
            p.requires_grad = False

        actor_params = [p for n, p in self.actor.named_parameters()
                        if 'importance' not in n]
        self.actor_opt = torch.optim.Adam(actor_params, lr=lr)
        self.critic_opt = torch.optim.Adam(self.critic.parameters(), lr=lr)
        if self.use_compression:
            self.imp_opt = torch.optim.Adam(self.actor.importance.parameters(), lr=0.01)

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

    @property
    def alpha(self):
        return self.log_alpha.exp().item()

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
        if self.use_compression:
            al = al + self.gamma_comp * self.actor.compression_loss()
        if self.use_ewc and self.ewc_fisher:
            ewc_loss = sum(
                (self.ewc_fisher[n] * (p - self.ewc_params[n]).pow(2)).sum()
                for n, p in self.actor.named_parameters() if n in self.ewc_fisher
            )
            al = al + self.cl_reg_coef * ewc_loss
        self.actor_opt.zero_grad()
        if self.use_compression:
            self.imp_opt.zero_grad()
        al.backward()
        # Gradient scaling by accumulated importance (soft protection)
        if self.use_compression and self.grad_scale_beta > 0:
            self._scale_actor_grads()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=10.0)
        self.actor_opt.step()
        if self.use_compression:
            self.imp_opt.step()

        # Alpha
        alpha_loss = -(self.log_alpha.exp() * (lp2 + self.target_entropy).detach()).mean()
        self.alpha_opt.zero_grad()
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

    def on_task_end(self, buffer):
        """Called after each task. Snapshot state for CL methods."""
        if self.use_replay:
            self.replay_store.add(buffer, n=5000)
        if self.use_compression:
            # Snapshot importance (accumulate max across tasks)
            with torch.no_grad():
                for i, imp in enumerate(self.actor.importance):
                    self.accumulated_importance[i] = torch.max(
                        self.accumulated_importance[i], imp.data.clamp(min=0))
        if self.use_ewc:
            # Compute Fisher on last task's data
            self._compute_ewc_fisher(buffer)

    def on_task_start(self, task_idx):
        """Called at start of each new task. Reset alpha to encourage exploration."""
        if task_idx > 0:
            # Reset log_alpha to restore exploration
            with torch.no_grad():
                self.log_alpha.fill_(0.0)  # alpha = 1.0
            # Rebuild alpha optimizer to reset its state
            self.alpha_opt = torch.optim.Adam([self.log_alpha], lr=3e-4)

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


# ========================================================================
# Evaluate on a task (deterministic)
# ========================================================================
@torch.no_grad()
def evaluate_task(agent, task_name, n_eval_envs=16):
    eval_env = make_env(task_name, n_envs=n_eval_envs)
    obs = eval_env.reset()
    success = torch.zeros(n_eval_envs, device=agent.device)
    for _ in range(MAX_EP_LEN):
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
def train_cl(method, tasks, steps_per_task=30_000, n_envs=64,
             start_steps=1000, update_freq=1.0, seed=42):
    """Run continual learning training over a sequence of tasks.

    Args:
        method: 'finetune', 'ewc', 'replay', 'compression', 'compression_replay'
        tasks: list of task names (e.g., ['reach-front', 'reach-top', ...])
        steps_per_task: env steps per task
        n_envs: parallel envs (higher = faster data collection)
        update_freq: gradient updates per env step (UTD ratio, lower = faster)
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    agent = SACAgentCL(method=method)
    buffer = ReplayBuffer(capacity=200_000)

    # Logged data: matrix[task_i_trained, task_j_evaluated] = success rate
    # plus learning curves
    n = len(tasks)
    eval_matrix = np.zeros((n, n))  # after training on task i, eval on all
    learning_curves = {t: [] for t in tasks}  # list of (step, success) per task

    t_start = time.time()

    for task_idx, task_name in enumerate(tasks):
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}/{n-1}: {task_name}")
        print(f"{'='*60}", flush=True)

        env = make_env(task_name, n_envs=n_envs)
        obs = env.reset()

        # Clear buffer per task (CW convention)
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

            # Log learning curve
            if task_steps % (n_envs * 40) < n_envs:
                elapsed = time.time() - task_t0
                sps = task_steps / max(elapsed, 1e-6)
                curr_succ = info['success_once'].float().mean().item()
                learning_curves[task_name].append((task_idx, task_steps, curr_succ))
                print(f'  [{task_steps//1000:4d}K/{steps_per_task//1000}K] '
                      f'sps={sps:.0f} succ_curr={curr_succ:.3f} '
                      f'alpha={agent.alpha:.3f}', flush=True)

        # End of task: snapshot for CL methods
        agent.on_task_end(buffer)

        # Evaluate on ALL tasks (for eval matrix)
        print(f'  Evaluating on all {n} tasks...', flush=True)
        for eval_idx, eval_task in enumerate(tasks):
            succ = evaluate_task(agent, eval_task, n_eval_envs=16)
            eval_matrix[task_idx, eval_idx] = succ
            print(f'    {eval_task:20s}: {succ:.3f}', flush=True)

        env.close()

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
                        choices=['finetune', 'ewc', 'replay', 'compression',
                                 'compression_replay'])
    parser.add_argument('--tasks', type=str, default='reach_cycle',
                        help='Task sequence: reach_cycle or comma-separated list')
    parser.add_argument('--steps_per_task', type=int, default=30_000)
    parser.add_argument('--n_envs', type=int, default=64)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--out', type=str, default='cl_result.json')
    args = parser.parse_args()

    if args.tasks == 'reach_cycle':
        tasks = ['reach-front', 'reach-top', 'reach-left', 'reach-right']
    elif args.tasks == 'reach_long':
        tasks = ['reach-front', 'reach-top', 'reach-left', 'reach-right',
                 'reach-front', 'reach-top', 'reach-left', 'reach-right']
    elif args.tasks == 'cw_subset':
        # Subset of real CW tasks that run on our GPU port
        tasks = ['push', 'window-close', 'faucet-close', 'handle-press-side']
    elif args.tasks == 'cw_full':
        tasks = ['hammer', 'push-wall', 'faucet-close', 'push-back',
                 'handle-press-side', 'push', 'window-close']
    else:
        tasks = args.tasks.split(',')

    print(f'CL experiment: method={args.method}, tasks={tasks}')
    print(f'steps/task={args.steps_per_task}, n_envs={args.n_envs}, seed={args.seed}')

    result = train_cl(
        method=args.method,
        tasks=tasks,
        steps_per_task=args.steps_per_task,
        n_envs=args.n_envs,
        seed=args.seed,
    )

    os.makedirs('checkpoints', exist_ok=True)
    with open(f'checkpoints/{args.out}', 'w') as f:
        json.dump(result, f, indent=2, default=str)
    print(f'Saved to checkpoints/{args.out}')


if __name__ == '__main__':
    main()
