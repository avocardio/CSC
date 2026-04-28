"""Compute learning-paradigm metrics from existing accuracy matrices:
  - Plasticity (diagonal trajectory): A[k][k] across k. Loss of plasticity
    shows up as a flat or declining trend.
  - Stability per task: A[T-1][j] - A[j][j] (negative = forgetting).
  - Final-vs-fresh gap: average final acc on first half of tasks vs latter half.

Run after pulling JSONs. Filters CIFAR-100 R18 by default."""
from __future__ import annotations
import os, sys, json, glob, argparse
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import numpy as np
from collections import defaultdict


def load(ckpt_dir, dataset='cifar100', num_tasks=10, model='resnet18'):
    by = defaultdict(list)
    default = 'mlp' if dataset == 'pmnist' else 'resnet18'
    for f in sorted(glob.glob(os.path.join(ckpt_dir, 'sup_*.json'))):
        d = json.load(open(f))
        cfg = d['config']
        if cfg.get('dataset', 'cifar100') != dataset: continue
        if cfg.get('num_tasks') != num_tasks: continue
        if dataset != 'pmnist' and cfg.get('model', default) != model: continue
        if cfg.get('tag', ''): continue
        by[cfg['method']].append(np.array(d['accuracy_matrix']) * 100)
    return {m: np.stack(ms) for m, ms in by.items() if ms}


def plasticity_curve(mats):
    """Diagonal A[k][k] mean over seeds, k=0..T-1.
    Higher = task at training time was learned well; if it declines for
    later tasks, the model is losing plasticity."""
    return mats[:, np.arange(mats.shape[1]), np.arange(mats.shape[1])].mean(0)


def stability_per_task(mats):
    """A[T-1][j] - A[j][j] for each j. Mean over seeds.
    0 = perfect retention. Negative = forgetting."""
    T = mats.shape[1]
    return (mats[:, T-1, :T] - mats[:, np.arange(T), np.arange(T)]).mean(0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--ckpt_dir', default='checkpoints')
    p.add_argument('--num_tasks', type=int, default=10)
    p.add_argument('--dataset', default='cifar100')
    args = p.parse_args()

    mats = load(args.ckpt_dir, args.dataset, args.num_tasks)
    if not mats:
        print('No matching JSONs.'); return
    print(f'\n=== Plasticity (diag A[k][k]; higher = preserved learning capacity) ===')
    print(f'{"method":12s} | ' + ' '.join(f'T{k}' for k in range(args.num_tasks)))
    print('-' * 72)
    plasticity = {}
    for m in ['finetune', 'replay', 'der', 'csc', 'ewc', 'mas', 'csc_ewc', 'csc_bd_ewc']:
        if m not in mats: continue
        c = plasticity_curve(mats[m])
        plasticity[m] = c
        print(f'{m:12s} | ' + ' '.join(f'{v:4.1f}' for v in c)
              + f'   mean={c.mean():.2f}   T0={c[0]:.2f} T-1={c[-1]:.2f}')

    print(f'\n=== Stability A[T-1][j] - A[j][j] per j (closer to 0 = less forgetting) ===')
    print(f'{"method":12s} | ' + ' '.join(f'T{k}' for k in range(args.num_tasks)))
    print('-' * 72)
    for m in ['finetune', 'replay', 'der', 'csc', 'ewc', 'mas', 'csc_ewc', 'csc_bd_ewc']:
        if m not in mats: continue
        s = stability_per_task(mats[m])
        print(f'{m:12s} | ' + ' '.join(f'{v:+5.1f}' for v in s)
              + f'   mean={s.mean():+.2f}')

    # Plasticity at last task vs first task — does the model still learn?
    print(f'\n=== Late-task plasticity (mean A[k][k] for k in last 5 tasks) ===')
    for m in ['finetune', 'replay', 'der', 'csc', 'ewc', 'mas', 'csc_ewc', 'csc_bd_ewc']:
        if m not in mats: continue
        c = plasticity_curve(mats[m])
        print(f'{m:12s}  early={c[:args.num_tasks//2].mean():.2f}  '
              f'late={c[args.num_tasks//2:].mean():.2f}  '
              f'gain={c[args.num_tasks//2:].mean() - c[:args.num_tasks//2].mean():+.2f}')


if __name__ == '__main__':
    main()
