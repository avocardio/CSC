"""Run all CL methods sequentially and collect results.

Usage: python rl/run_all_methods.py --tasks reach_cycle --steps 30000
"""

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import argparse
import json
import time
import numpy as np

from rl.cl_experiment import train_cl


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--tasks', type=str, default='reach_cycle')
    parser.add_argument('--steps', type=int, default=30000)
    parser.add_argument('--n_envs', type=int, default=128)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--methods', type=str,
                        default='finetune,ewc,replay,compression,csc')
    args = parser.parse_args()

    if args.tasks == 'reach_cycle':
        tasks = ['reach-front', 'reach-top', 'reach-left', 'reach-right']
    elif args.tasks == 'cw_subset':
        tasks = ['push', 'window-close', 'faucet-close', 'handle-press-side']
    else:
        tasks = args.tasks.split(',')

    methods = args.methods.split(',')

    results = {}
    t_all = time.time()

    for method in methods:
        print(f"\n{'#'*70}")
        print(f"# METHOD: {method}")
        print(f"{'#'*70}\n", flush=True)
        t0 = time.time()
        try:
            res = train_cl(
                method=method,
                tasks=tasks,
                steps_per_task=args.steps,
                n_envs=args.n_envs,
                seed=args.seed,
            )
            res['wall_time'] = time.time() - t0
            results[method] = res
            print(f"\n{method} finished in {res['wall_time']:.0f}s")
        except Exception as e:
            print(f"\n{method} FAILED: {e}")
            import traceback
            traceback.print_exc()
            results[method] = {'error': str(e)}

    total = time.time() - t_all

    # Comparison table
    print(f"\n{'='*70}")
    print(f"SUMMARY ({total:.0f}s total)")
    print(f"{'='*70}")
    print(f"{'method':<25} {'avg_perf':<10} {'forget':<10} {'bwt':<10} {'fwt':<10}")
    for m, res in results.items():
        if 'metrics' in res:
            met = res['metrics']
            print(f"{m:<25} "
                  f"{met['avg_performance']:<10.3f} "
                  f"{met['avg_forgetting']:<10.3f} "
                  f"{met['backward_transfer']:<10.3f} "
                  f"{met['forward_transfer']:<10.3f}")
        else:
            print(f"{m:<25} ERROR: {res.get('error', '?')}")

    os.makedirs('checkpoints', exist_ok=True)
    out_file = f'checkpoints/cl_all_methods_{args.tasks}_s{args.seed}.json'
    with open(out_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\nSaved to {out_file}")


if __name__ == '__main__':
    main()
