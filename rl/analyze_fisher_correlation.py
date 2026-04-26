"""Fisher correlation analysis for CSC RL.

For a CSC checkpoint trained on N tasks, compute:
1. Per-channel learned bit-depth (after training)
2. Per-channel Fisher information (analytic Gaussian Fisher, summed across channels)
3. Spearman rank correlation between the two

This is the RL analog of the supervised paper's bit-depth ↔ Fisher correlation
analysis (rho=0.71). The claim is that compression discovers parameter
importance as a free byproduct.

Usage:
    python rl/analyze_fisher_correlation.py --ckpt checkpoints/cw_csc_cw10_s42.pt
"""
from __future__ import annotations
import os, sys, argparse, json
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import numpy as np
from scipy import stats

from rl.continual_world import (
    SACAgent, ReplayBuffer, make_env, load_checkpoint, CW10_TASKS,
    OBS_DIM, ACT_DIM, MAX_EP_LEN, HIDDEN, N_CORE_LAYERS,
)


def collect_obs(task_name: str, n_obs: int = 2560,
                actor=None, task_idx: int = 0) -> torch.Tensor:
    """Collect observations from a task using the trained actor (or random)."""
    env = make_env(task_name)
    out = []
    obs, _ = env.reset()
    ep_len = 0
    while len(out) < n_obs:
        out.append(obs.copy())
        if actor is not None:
            a = actor.act_stochastic(obs, task_idx)
        else:
            a = env.action_space.sample()
        obs, _, term, trunc, _ = env.step(a)
        ep_len += 1
        if term or trunc or ep_len >= MAX_EP_LEN:
            obs, _ = env.reset()
            ep_len = 0
    return torch.tensor(np.stack(out[:n_obs]), dtype=torch.float32)


def analytic_fisher(actor, obs: torch.Tensor, task_idx: int,
                     bs: int = 256) -> dict[str, torch.Tensor]:
    """Analytic Gaussian Fisher (CW reference): per-sample, per-output-dim,
    aggregated over batch. Returns dict[name -> Fisher tensor matching param shape].
    """
    from torch.func import functional_call, vmap, grad

    params = {n: p.detach() for n, p in actor.named_parameters()}
    names = [n for n in params if 'quantizer' not in n]
    fisher = {n: torch.zeros_like(params[n]) for n in names}

    def f_mu_j(p_dict, sample, j):
        mu, _ = functional_call(actor, p_dict,
                                (sample.unsqueeze(0), int(task_idx)))
        return mu.squeeze(0)[j]

    def f_ls_j(p_dict, sample, j):
        _, log_std = functional_call(actor, p_dict,
                                     (sample.unsqueeze(0), int(task_idx)))
        return log_std.squeeze(0)[j]

    n_total = obs.shape[0]
    n_batches = (n_total + bs - 1) // bs
    for b in range(n_batches):
        s = obs[b * bs:(b + 1) * bs]
        with torch.no_grad():
            _, log_std_b = actor(s, task_idx)
            std_b = log_std_b.exp().clamp(min=1e-3)
        for j in range(ACT_DIM):
            g_mu = vmap(grad(f_mu_j), in_dims=(None, 0, None))(params, s, j)
            g_ls = vmap(grad(f_ls_j), in_dims=(None, 0, None))(params, s, j)
            sj = std_b[:, j]
            for n in names:
                if n not in g_mu:
                    continue
                mg = g_mu[n]
                lg = g_ls[n]
                extra = mg.dim() - 1
                sb = sj.view(-1, *([1] * extra))
                sg = sb * lg
                f = (mg.pow(2) + 2 * sg.pow(2)) / (sb.pow(2) + 1e-6)
                fisher[n] = fisher[n] + f.sum(0)
    for n in names:
        fisher[n] = fisher[n] / n_total
    return fisher


def per_channel_fisher_for_layer(fisher: dict[str, torch.Tensor],
                                 layer_name: str, hidden: int = HIDDEN) -> torch.Tensor:
    """Aggregate Fisher across the input dimension to get per-output-channel value.

    For a Linear layer of shape (out, in), Fisher is shape (out, in). Per output
    channel: sum across input dim. Plus add Fisher of the corresponding bias.
    """
    weight_key = f'{layer_name}.weight'
    bias_key = f'{layer_name}.bias'
    if weight_key not in fisher:
        return None
    F_w = fisher[weight_key]                  # (out, in)
    F_b = fisher.get(bias_key, torch.zeros(F_w.shape[0]))
    per_channel = F_w.sum(dim=1) + F_b
    return per_channel


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt', required=True, help='path to CSC checkpoint')
    p.add_argument('--task_idx', type=int, default=None,
                   help='task index for evaluation (default: last completed)')
    p.add_argument('--n_obs', type=int, default=2560)
    p.add_argument('--out', default='', help='output json path (optional)')
    args = p.parse_args()

    if not os.path.isfile(args.ckpt):
        print(f'Checkpoint not found: {args.ckpt}'); return

    # Build agent + load
    raw = torch.load(args.ckpt, map_location='cpu', weights_only=False)
    method = raw['method']
    if method != 'csc':
        print(f'Warning: ckpt method={method}, not csc. Bit-depths may be uninformative.')
    n_tasks = raw['log_alpha'].shape[0]

    agent = SACAgent(method=method, n_tasks=n_tasks, device='cpu')
    task_idx, eval_history, results, rng = load_checkpoint(args.ckpt, agent)
    if args.task_idx is None:
        args.task_idx = task_idx
    print(f'Loaded ckpt: method={method} last_task={task_idx} '
          f'analyzing task_idx={args.task_idx}')

    actor = agent.actor.eval()
    task_name = CW10_TASKS[args.task_idx] if args.task_idx < len(CW10_TASKS) else CW10_TASKS[args.task_idx % 10]

    print(f'Collecting {args.n_obs} obs from {task_name}...')
    obs = collect_obs(task_name, n_obs=args.n_obs, actor=actor, task_idx=args.task_idx)
    print(f'Computing Fisher information...')
    fisher = analytic_fisher(actor, obs, args.task_idx)

    # Per-channel Fisher and bit-depth for each core layer
    layer_names = ['fc1', 'fc2', 'fc3', 'fc4']
    per_layer = {}
    all_b = []
    all_f = []
    for ln in layer_names:
        b = getattr(actor, ln).channel_bit_depths().detach().cpu().numpy()
        F_per = per_channel_fisher_for_layer(fisher, ln)
        if F_per is None:
            continue
        F_per = F_per.detach().cpu().numpy()
        rho, p_val = stats.spearmanr(b, F_per)
        per_layer[ln] = {'spearman': float(rho), 'p': float(p_val),
                         'bit_mean': float(b.mean()), 'bit_std': float(b.std()),
                         'fisher_mean': float(F_per.mean()),
                         'fisher_std': float(F_per.std())}
        all_b.append(b); all_f.append(F_per)
        print(f'  {ln}: rho={rho:+.3f} (p={p_val:.2e}) '
              f'b={b.mean():.2f}±{b.std():.2f} '
              f'F={F_per.mean():.2e}±{F_per.std():.2e}')

    # Overall correlation
    all_b = np.concatenate(all_b); all_f = np.concatenate(all_f)
    rho_all, p_all = stats.spearmanr(all_b, all_f)
    print(f'\nOverall Spearman rho: {rho_all:+.3f} (p={p_all:.2e})')

    out = {
        'ckpt': args.ckpt,
        'method': method,
        'task_idx': args.task_idx,
        'task_name': task_name,
        'n_obs': args.n_obs,
        'per_layer': per_layer,
        'overall_spearman': float(rho_all),
        'overall_p': float(p_all),
    }
    if args.out:
        with open(args.out, 'w') as f:
            json.dump(out, f, indent=2)
        print(f'Saved: {args.out}')


if __name__ == '__main__':
    main()
