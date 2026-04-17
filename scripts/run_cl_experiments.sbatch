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

echo "Starting CL experiments at $(date)"
echo "Node: $(hostname), GPUs: $(nvidia-smi -L | wc -l)"

# Run 4 experiments in parallel, one per GPU
# GPU 0: finetune (baseline)
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune \
    --tasks cw_subset \
    --steps_per_task 50000 \
    --n_envs 64 \
    --seed 42 \
    --out results/cl_finetune_cw_s42.json \
    > results/finetune_s42.log 2>&1 &

# GPU 1: replay
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method replay \
    --tasks cw_subset \
    --steps_per_task 50000 \
    --n_envs 64 \
    --seed 42 \
    --out results/cl_replay_cw_s42.json \
    > results/replay_s42.log 2>&1 &

# GPU 2: ewc
CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method ewc \
    --tasks cw_subset \
    --steps_per_task 50000 \
    --n_envs 64 \
    --seed 42 \
    --out results/cl_ewc_cw_s42.json \
    > results/ewc_s42.log 2>&1 &

# GPU 3: compression_replay (ours)
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method compression_replay \
    --tasks cw_subset \
    --steps_per_task 50000 \
    --n_envs 64 \
    --seed 42 \
    --out results/cl_comp_replay_cw_s42.json \
    > results/comp_replay_s42.log 2>&1 &

# Wait for all
wait

echo "All experiments done at $(date)"
echo "=== RESULTS ==="
for f in results/cl_*_cw_s42.json; do
    method=$(basename $f | sed "s/cl_\(.*\)_cw_s42.json/\1/")
    if [ -f "$f" ]; then
        echo "--- $method ---"
        python -c "import json; d=json.load(open(\"$f\")); m=d[\"metrics\"]; print(f\"  avg_perf={m[\"avg_performance\"]:.3f}  forget={m[\"avg_forgetting\"]:.3f}  bwt={m[\"backward_transfer\"]:.3f}  fwt={m[\"forward_transfer\"]:.3f}\")"
    fi
done
'
