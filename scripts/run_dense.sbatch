#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -J csc_dense
#SBATCH -o scripts/dense.%j.out
#SBATCH -e scripts/dense.%j.err

# DENSE packing: multiple experiments per GPU.
# Each experiment uses ~1.3GB VRAM. GH200 has 97GB.
# We can fit 3-4 experiments per GPU easily.
#
# Layout: 4 GPUs × 3 experiments = 12 total
# GPU 0: finetune s42, finetune s1, finetune s2
# GPU 1: replay s42, replay s1, replay s2
# GPU 2: ewc s42, ewc s1, ewc s2
# GPU 3: csc s42, csc s1, csc s2
#
# This runs ALL 12 experiments simultaneously!

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

echo "DENSE experiment at $(date)"
echo "Node: $(hostname), 12 experiments on 4 GPUs (3 per GPU)"

# GPU 0: finetune × 3 seeds
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/dense_finetune_s${SEED}.json \
        > results/dense_finetune_s${SEED}.log 2>&1 &
done

# GPU 1: replay × 3 seeds
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method replay --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/dense_replay_s${SEED}.json \
        > results/dense_replay_s${SEED}.log 2>&1 &
done

# GPU 2: ewc × 3 seeds
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method ewc --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/dense_ewc_s${SEED}.json \
        > results/dense_ewc_s${SEED}.log 2>&1 &
done

# GPU 3: csc × 3 seeds
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
        --method csc --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/dense_csc_s${SEED}.json \
        > results/dense_csc_s${SEED}.log 2>&1 &
done

wait

echo ""
echo "=== ALL RESULTS (DENSE) ==="
for method in finetune replay ewc csc; do
    echo "--- $method ---"
    for s in 42 1 2; do
        echo -n "  s$s: "
        grep "avg_performance\|avg_forgetting" results/dense_${method}_s${s}.log 2>/dev/null | tr "\n" " " || echo "missing"
        echo
    done
done
echo "Done at $(date)"
'
