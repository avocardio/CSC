#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J csc_cw
#SBATCH -o scripts/cw_proper.%j.out
#SBATCH -e scripts/cw_proper.%j.err

# Continual World experiments with CORRECT hyperparameters:
# - n_envs=256 (moderate parallelism, not too high for SAC)
# - UTD=1.0 (matching CW paper)
# - lr=1e-3, batch=128 (matching CW paper)
# - 1M steps/task (matching CW paper)
# - 4 tasks: window-close, faucet-close, handle-press-side, peg-unplug-side

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail

module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

export TORCH_HOME=/capstor/scratch/cscs/$USER/CSC/.cache/torch
export TRITON_CACHE_DIR=/capstor/scratch/cscs/$USER/CSC/.cache/triton
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"
mkdir -p results

TASKS=cw_learnable
STEPS=250000
ENVS=256
SEED=42

echo "CW Proper experiments at $(date)"
echo "Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l)"
echo "Config: tasks=$TASKS steps=$STEPS n_envs=$ENVS seed=$SEED UTD=1.0"
echo "SAC hyperparams: lr=1e-3 batch=128 buffer=1M (matching CW paper)"

# 4 methods in parallel, one per GPU
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cw_finetune_s${SEED}.json \
    > results/cw_finetune_s${SEED}.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method replay --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cw_replay_s${SEED}.json \
    > results/cw_replay_s${SEED}.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method ewc --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cw_ewc_s${SEED}.json \
    > results/cw_ewc_s${SEED}.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method csc --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cw_csc_s${SEED}.json \
    > results/cw_csc_s${SEED}.log 2>&1 &

wait

echo ""
echo "All CW experiments done at $(date)"
echo "=== RESULTS ==="
for method in finetune replay ewc csc; do
    echo "--- $method ---"
    tail -10 "results/cw_${method}_s${SEED}.log" 2>/dev/null || echo "  no log"
done
'
