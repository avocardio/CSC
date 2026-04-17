"""Continual RL on Meta-World tasks (ContinualWorld-style).

Sequences 10 robotic manipulation tasks, each trained for a fixed number of steps.
Compares:
- Fine-tuning (just keep training, catastrophic forgetting)
- EWC on the policy network
- Self-compression guided unit replacement
- PackNet-style pruning and freezing

This is where self-compression should shine: RL has non-stationary data within
each task (policy changes → visited states change), creating ongoing representation
pressure that compression can manage dynamically.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import argparse
import time
import metaworld
import random as py_random

from rl.sac import SAC, ReplayBuffer


def make_metaworld_env(task_name, seed=0):
    """Create a Meta-World environment for a given task."""
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    task = py_random.choice(ml1.train_tasks)
    env.set_task(task)
    return env


def evaluate_task(agent, task_name, n_episodes=10, seed=0):
    """Evaluate agent on a specific task."""
    env = make_metaworld_env(task_name, seed)
    total_reward = 0
    total_success = 0

    for ep in range(n_episodes):
        obs, _ = env.reset()
        episode_reward = 0
        for step in range(500):  # Meta-World max steps
            action = agent.select_action(obs, evaluate=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            if terminated or truncated:
                break
        total_reward += episode_reward
        total_success += float(info.get('success', 0))

    env.close()
    return total_reward / n_episodes, total_success / n_episodes


def train_continual_metaworld(config, device='cuda'):
    """Train SAC sequentially on Meta-World tasks."""
    task_names = [
        'reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'drawer-close-v3',
        'button-press-topdown-v3', 'peg-insert-side-v3', 'window-open-v3',
        'sweep-v3', 'basketball-v3',
    ]

    num_tasks = config.get('num_tasks', len(task_names))
    task_names = task_names[:num_tasks]
    steps_per_task = config.get('steps_per_task', 500_000)
    eval_interval = config.get('eval_interval', 50_000)
    batch_size = config.get('batch_size', 256)
    start_steps = config.get('start_steps', 5000)

    # Create agent
    obs_dim = 39  # Meta-World observation dimension
    act_dim = 4   # Meta-World action dimension
    agent = SAC(obs_dim, act_dim, hidden_dim=256, device=device)

    replay_buffer = ReplayBuffer(capacity=1_000_000)

    all_results = []

    for task_idx, task_name in enumerate(task_names):
        print(f"\n{'='*60}")
        print(f"TASK {task_idx}: {task_name} ({steps_per_task} steps)")
        print(f"{'='*60}")

        env = make_metaworld_env(task_name)
        obs, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        episode_num = 0

        for step in range(steps_per_task):
            # Select action
            if step < start_steps and task_idx == 0:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            replay_buffer.push(obs, action, reward, next_obs, float(done))

            obs = next_obs
            episode_reward += reward
            episode_steps += 1

            if done or episode_steps >= 500:
                obs, _ = env.reset()
                episode_num += 1
                episode_reward = 0
                episode_steps = 0

            # Update agent
            if len(replay_buffer) >= batch_size:
                agent.update(replay_buffer, batch_size)

            # Evaluate periodically
            if (step + 1) % eval_interval == 0:
                # Evaluate on current task
                avg_reward, success_rate = evaluate_task(agent, task_name)
                print(f"  Step {step+1}: reward={avg_reward:.1f}, success={success_rate:.2f}")

                # Evaluate on all previous tasks
                task_results = {}
                for prev_idx in range(task_idx + 1):
                    prev_name = task_names[prev_idx]
                    prev_reward, prev_success = evaluate_task(agent, prev_name, n_episodes=5)
                    task_results[prev_name] = {'reward': prev_reward, 'success': prev_success}

                all_results.append({
                    'task_idx': task_idx,
                    'step': step + 1,
                    'global_step': task_idx * steps_per_task + step + 1,
                    'task_results': task_results,
                })

        env.close()

        # End-of-task evaluation on all tasks
        print(f"\nEnd of {task_name}:")
        for prev_idx in range(task_idx + 1):
            prev_name = task_names[prev_idx]
            prev_reward, prev_success = evaluate_task(agent, prev_name)
            print(f"  {prev_name}: reward={prev_reward:.1f}, success={prev_success:.2f}")

    return all_results, agent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--steps_per_task', type=int, default=200_000)
    parser.add_argument('--eval_interval', type=int, default=50_000)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_random.seed(args.seed)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    config = vars(args)

    print(f"Continual Meta-World: {args.num_tasks} tasks, {args.steps_per_task} steps/task")
    results, agent = train_continual_metaworld(config, device)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'results': results, 'config': config},
               f'checkpoints/continual_metaworld_t{args.num_tasks}.pt')


if __name__ == '__main__':
    main()
