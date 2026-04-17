"""Run continual RL experiments comparing standard SAC vs compression SAC."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
import random as py_random
import argparse
import time
import metaworld

from rl.sac import SAC, ReplayBuffer
from rl.compression_sac import CompressionSAC


def make_env(task_name):
    ml1 = metaworld.ML1(task_name)
    env = ml1.train_classes[task_name]()
    task = py_random.choice(ml1.train_tasks)
    env.set_task(task)
    return env


def evaluate(agent, task_name, n_episodes=10):
    env = make_env(task_name)
    total_reward, total_success = 0, 0
    for _ in range(n_episodes):
        obs, _ = env.reset()
        ep_reward = 0
        for _ in range(500):
            action = agent.select_action(obs, evaluate=True)
            obs, reward, term, trunc, info = env.step(action)
            ep_reward += reward
            if term or trunc: break
        total_reward += ep_reward
        total_success += float(info.get('success', 0))
    env.close()
    return total_reward / n_episodes, total_success / n_episodes


def train_sequential(agent_type, config, device='cuda'):
    tasks = ['reach-v3', 'push-v3', 'pick-place-v3', 'door-open-v3', 'drawer-close-v3']
    num_tasks = min(config['num_tasks'], len(tasks))
    tasks = tasks[:num_tasks]
    steps_per_task = config['steps_per_task']

    obs_dim, act_dim = 39, 4

    if agent_type == 'standard':
        agent = SAC(obs_dim, act_dim, hidden_dim=256, device=device)
    elif agent_type == 'compression':
        agent = CompressionSAC(obs_dim, act_dim, hidden_dim=256, device=device,
                               gamma_comp=config.get('gamma', 0.01),
                               replacement_rate=config.get('replacement_rate', 0.001))
    else:
        raise ValueError(f"Unknown agent type: {agent_type}")

    replay_buf = ReplayBuffer(capacity=1_000_000)
    results = []

    for task_idx, task_name in enumerate(tasks):
        print(f"\n=== TASK {task_idx}: {task_name} ===")
        env = make_env(task_name)
        obs, _ = env.reset()
        ep_steps = 0

        for step in range(steps_per_task):
            if step < 5000 and task_idx == 0:
                action = env.action_space.sample()
            else:
                action = agent.select_action(obs)

            next_obs, reward, term, trunc, info = env.step(action)
            replay_buf.push(obs, action, reward, next_obs, float(term or trunc))
            obs = next_obs
            ep_steps += 1

            if term or trunc or ep_steps >= 500:
                obs, _ = env.reset()
                ep_steps = 0

            if len(replay_buf) >= 256:
                agent.update(replay_buf, 256)

            if (step + 1) % 50000 == 0:
                avg_r, succ = evaluate(agent, task_name, n_episodes=5)
                # Also evaluate previous tasks
                prev_results = {}
                for pi in range(task_idx + 1):
                    pr, ps = evaluate(agent, tasks[pi], n_episodes=5)
                    prev_results[tasks[pi]] = {'reward': pr, 'success': ps}

                replaced = getattr(agent, 'total_replaced', 0)
                print(f"  Step {step+1}: {task_name} reward={avg_r:.0f} success={succ:.2f} replaced={replaced}")
                for pn, pv in prev_results.items():
                    if pn != task_name:
                        print(f"    {pn}: reward={pv['reward']:.0f} success={pv['success']:.2f}")

                results.append({
                    'task_idx': task_idx, 'step': step+1,
                    'current': {'reward': avg_r, 'success': succ},
                    'all_tasks': prev_results,
                })

        env.close()

    # Final evaluation on all tasks
    print(f"\n=== FINAL EVALUATION ===")
    final = {}
    for t in tasks:
        r, s = evaluate(agent, t, n_episodes=10)
        final[t] = {'reward': r, 'success': s}
        print(f"  {t}: reward={r:.0f}, success={s:.2f}")

    avg_success = np.mean([v['success'] for v in final.values()])
    print(f"  Average success: {avg_success:.2f}")

    return results, final


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--agent', type=str, required=True, choices=['standard', 'compression'])
    parser.add_argument('--num_tasks', type=int, default=5)
    parser.add_argument('--steps_per_task', type=int, default=200_000)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--replacement_rate', type=float, default=0.001)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    py_random.seed(args.seed)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'

    print(f"Continual RL: {args.agent}, {args.num_tasks} tasks, {args.steps_per_task} steps/task")

    results, final = train_sequential(args.agent, vars(args), device)

    os.makedirs('checkpoints', exist_ok=True)
    torch.save({'results': results, 'final': final, 'config': vars(args)},
               f'checkpoints/continual_rl_{args.agent}.pt')


if __name__ == '__main__':
    main()
