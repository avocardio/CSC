"""Collect CW10 benchmark results into a summary table."""
import json
import os
import numpy as np

METHODS = ['finetune', 'ewc', 'replay', 'packnet', 'csc']
SEEDS = [42, 123, 456, 789, 1337]
METRICS = ['avg_performance', 'avg_forgetting', 'backward_transfer', 'forward_transfer']

results = {m: {k: [] for k in METRICS} for m in METHODS}

for method in METHODS:
    for seed in SEEDS:
        path = f'checkpoints/bench_{method}_s{seed}.json'
        if not os.path.exists(path):
            continue
        with open(path) as f:
            data = json.load(f)
        for k in METRICS:
            if k in data.get('metrics', {}):
                results[method][k].append(data['metrics'][k])

print(f"{'Method':<12} {'Avg Perf':>12} {'Forgetting':>12} {'BWT':>12} {'FWT':>12} {'Seeds':>6}")
print('-' * 70)
for method in METHODS:
    n = len(results[method]['avg_performance'])
    if n == 0:
        print(f"{method:<12} {'(no data)':>12}")
        continue
    row = []
    for k in METRICS:
        vals = results[method][k]
        if vals:
            row.append(f"{np.mean(vals):.3f}±{np.std(vals):.3f}")
        else:
            row.append("N/A")
    print(f"{method:<12} {row[0]:>12} {row[1]:>12} {row[2]:>12} {row[3]:>12} {n:>6}")
