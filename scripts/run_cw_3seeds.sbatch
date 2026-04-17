#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J csc_3s
#SBATCH -o scripts/cw_3seeds.%j.out
#SBATCH -e scripts/cw_3seeds.%j.err

# CW experiments with fixed eval, 3 seeds, CW paper hyperparams.
# Runs 4 methods x 3 seeds = 12 experiments, 4 at a time (one per GPU).

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export TORCH_HOME=/capstor/scratch/cscs/$USER/CSC/.cache/torch
export TRITON_CACHE_DIR=/capstor/scratch/cscs/$USER/CSC/.cache/triton
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR" results

TASKS=cw_learnable
STEPS=250000
ENVS=256

echo "CW 3-seed experiments at $(date)"
echo "Node: $(hostname)"

for SEED in 42 1 2; do
    echo ""
    echo "========== SEED $SEED =========="

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
    echo "Seed $SEED done at $(date)"
done

echo ""
echo "=== ALL RESULTS ==="
for method in finetune replay ewc csc; do
    echo "--- $method ---"
    for s in 42 1 2; do
        echo -n "  s$s: "
        grep "avg_performance\|avg_forgetting" results/cw_${method}_s${s}.log 2>/dev/null | tr "\n" " " || echo "missing"
        echo
    done
done
echo "Done at $(date)"
'
