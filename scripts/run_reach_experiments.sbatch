#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J csc_reach
#SBATCH -o scripts/reach_experiments.%j.out
#SBATCH -e scripts/reach_experiments.%j.err

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

TASKS=reach_cycle
STEPS=30000
ENVS=1024
SEED=42

echo "Starting reach CL experiments at $(date)"
echo "Node: $(hostname)"
echo "Config: tasks=$TASKS steps=$STEPS n_envs=$ENVS seed=$SEED"

# GPU 0: finetune
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_finetune_${TASKS}_s${SEED}.json \
    > results/finetune_${TASKS}_s${SEED}.log 2>&1 &

# GPU 1: replay
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method replay --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_replay_${TASKS}_s${SEED}.json \
    > results/replay_${TASKS}_s${SEED}.log 2>&1 &

# GPU 2: ewc
CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method ewc --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_ewc_${TASKS}_s${SEED}.json \
    > results/ewc_${TASKS}_s${SEED}.log 2>&1 &

# GPU 3: compression_replay
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method compression_replay --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_comp_replay_${TASKS}_s${SEED}.json \
    > results/comp_replay_${TASKS}_s${SEED}.log 2>&1 &

wait
echo "All reach experiments done at $(date)"

# Also run seeds 1 and 2 for error bars
for SEED2 in 1 2; do
    echo "=== SEED $SEED2 ==="
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED2 \
        --out results/cl_finetune_${TASKS}_s${SEED2}.json \
        > results/finetune_${TASKS}_s${SEED2}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method replay --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED2 \
        --out results/cl_replay_${TASKS}_s${SEED2}.json \
        > results/replay_${TASKS}_s${SEED2}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method ewc --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED2 \
        --out results/cl_ewc_${TASKS}_s${SEED2}.json \
        > results/ewc_${TASKS}_s${SEED2}.log 2>&1 &

    CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
        --method compression_replay --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED2 \
        --out results/cl_comp_replay_${TASKS}_s${SEED2}.json \
        > results/comp_replay_${TASKS}_s${SEED2}.log 2>&1 &

    wait
    echo "Seed $SEED2 done at $(date)"
done

echo ""
echo "=== ALL RESULTS (3 seeds) ==="
for method in finetune replay ewc comp_replay; do
    echo "--- $method ---"
    for s in 42 1 2; do
        f="results/cl_${method}_${TASKS}_s${s}.json"
        if [ -f "$f" ]; then
            python -c "import json; d=json.load(open(\"$f\")); m=d[\"metrics\"]; print(f\"  s{s}: perf={m[\"avg_performance\"]:.3f} forget={m[\"avg_forgetting\"]:.3f} bwt={m[\"backward_transfer\"]:.3f} fwt={m[\"forward_transfer\"]:.3f}\")"
        fi
    done
done
'
