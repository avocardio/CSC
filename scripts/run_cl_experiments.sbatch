#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J csc_cl
#SBATCH -o scripts/cl_experiments.%j.out
#SBATCH -e scripts/cl_experiments.%j.err

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
STEPS=200000
ENVS=4096
SEED=42

echo "Starting CL experiments at $(date)"
echo "Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l)"
echo "Config: tasks=$TASKS steps=$STEPS n_envs=$ENVS seed=$SEED"

# Run 4 experiments in parallel, one per GPU
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_finetune_${TASKS}_s${SEED}.json \
    > results/finetune_${TASKS}_s${SEED}.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method replay --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_replay_${TASKS}_s${SEED}.json \
    > results/replay_${TASKS}_s${SEED}.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method ewc --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_ewc_${TASKS}_s${SEED}.json \
    > results/ewc_${TASKS}_s${SEED}.log 2>&1 &

CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method compression_replay --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed $SEED \
    --out results/cl_comp_replay_${TASKS}_s${SEED}.json \
    > results/comp_replay_${TASKS}_s${SEED}.log 2>&1 &

wait

echo "All experiments done at $(date)"
echo ""
echo "=== RESULTS ==="
for f in results/cl_*_${TASKS}_s${SEED}.json; do
    method=$(basename $f | sed "s/cl_\(.*\)_${TASKS}_s${SEED}.json/\1/")
    if [ -f "$f" ]; then
        echo "--- $method ---"
        python -c "import json; d=json.load(open(\"$f\")); m=d[\"metrics\"]; print(f\"  avg_perf={m[\"avg_performance\"]:.3f}  forget={m[\"avg_forgetting\"]:.3f}  bwt={m[\"backward_transfer\"]:.3f}  fwt={m[\"forward_transfer\"]:.3f}\")"
    else
        echo "--- $method --- MISSING"
    fi
done
'
