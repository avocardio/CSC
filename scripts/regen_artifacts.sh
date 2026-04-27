#!/bin/bash
# Re-generate all paper artifacts (plots + tables) from current checkpoint JSONs.
# Idempotent. Run after pulling new JSONs from the cluster.

set -euo pipefail
cd "$(dirname "$0")/.."

echo "=== Pulling latest JSONs from cluster ==="
eval "$(ssh-agent -s)" >/dev/null 2>&1
ssh-add ~/.ssh/cscs-key 2>/dev/null
ssh clariden "cd /capstor/scratch/cscs/mkalcher/CSC && tar c -C checkpoints \$(cd checkpoints && ls sup_*_final.json sup_*.json cw_*_final.json 2>/dev/null)" 2>/dev/null \
    | tar x -C checkpoints/ 2>/dev/null || true

echo
echo "=== Supervised: CIFAR-100 10-task curves + heatmap ==="
python analysis/cl_curves.py --num_tasks 10 --dataset cifar100 || true
python analysis/per_task_curves.py --num_tasks 10 --dataset cifar100 || true

echo
echo "=== Supervised: PMNIST 10-task curves + heatmap ==="
python analysis/cl_curves.py --num_tasks 10 --dataset pmnist || true
python analysis/per_task_curves.py --num_tasks 10 --dataset pmnist || true

echo
echo "=== Supervised: PMNIST 50-task curves ==="
python analysis/cl_curves.py --num_tasks 50 --dataset pmnist || true

echo
echo "=== Supervised: model-size scaling ==="
python analysis/scaling_plot.py || true

echo
echo "=== Supervised: tables ==="
python analysis/pareto_from_json.py 2>/dev/null || true

echo
echo "=== RL: ContinualWorld summary table ==="
python analysis/rl_summary.py

echo
echo "=== Done ==="
ls -la csc_paper/figures/*.pdf 2>/dev/null | tail -10
ls -la csc_paper/tables/*.tex 2>/dev/null | tail -5
