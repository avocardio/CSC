#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J cw10_val
#SBATCH -o scripts/cw10_val.%j.out
#SBATCH -e scripts/cw10_val.%j.err

# Validation: 1 seed, all 5 methods, full CW10 (10 tasks), 250K steps/task.
# GPU 0: finetune (then csc after it finishes)
# GPU 1: ewc
# GPU 2: replay
# GPU 3: packnet
# Each experiment: 10 tasks × 250K steps at ~80 sps ≈ 8.7h
# finetune + csc sequential on GPU 0 ≈ 17.4h — won't fit in 12h.
# So: run finetune/ewc/replay/packnet first (8.7h), then csc (8.7h) in a chained job.
# This script runs batch 1 (4 methods). Submit batch 2 (csc) with dependency.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "CW10 validation batch 1 at $(date)"
echo "4 methods: finetune, ewc, replay, packnet | seed=42, 250K steps/task"

CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks cw10 \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --seed 42 --out val_finetune_s42.json \
    > checkpoints/val_finetune_s42.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method ewc --tasks cw10 \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --seed 42 --out val_ewc_s42.json \
    > checkpoints/val_ewc_s42.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method replay --tasks cw10 \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --seed 42 --out val_replay_s42.json \
    > checkpoints/val_replay_s42.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method packnet --tasks cw10 \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --seed 42 --out val_packnet_s42.json \
    > checkpoints/val_packnet_s42.log 2>&1 &

wait

echo ""
echo "=========================================="
echo "CW10 VALIDATION BATCH 1 RESULTS"
echo "=========================================="
for method in finetune ewc replay packnet; do
    echo "--- $method ---"
    grep "avg_performance\|avg_forgetting\|backward_transfer\|forward_transfer\|DONE" \
        checkpoints/val_${method}_s42.log 2>/dev/null | tail -6
    echo
done
echo "Batch 1 done at $(date)"
'
