#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:0
#SBATCH -J rl_phases
#SBATCH -o scripts/rl_phases.%j.out
#SBATCH -e scripts/rl_phases.%j.err

# RL phases sweep:
#   shared_alpha replay + csc (Phase 1+2)
#   clonex + csc_clonex (Phase 3)
# Tasks: cw_subset (4-task subset, faster). 200k steps/task, 5 seeds each.
# 4 configs x 5 seeds = 20 parallel CPU runs.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

echo "RL phases sweep at $(date)"
SEEDS=(42 123 456 789 1234)
PIDS=()

# Phase 1+2: shared_alpha replay/csc
for method in replay csc; do
    for seed in "${SEEDS[@]}"; do
        tag="_shared"
        ckpt="checkpoints/cw_${method}_cw_subset_s${seed}${tag}.pt"
        log="checkpoints/cpu_${method}${tag}_s${seed}.log"
        echo "  ${method}+shared_alpha seed=${seed}"
        python rl/continual_world.py \
            --method "$method" --shared_alpha \
            --steps_per_task 200000 \
            --seed "$seed" --tasks cw_subset --tag "$tag" \
            --ckpt "$ckpt" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done
done

# Phase 3: clonex / csc_clonex
for method in clonex csc_clonex; do
    for seed in "${SEEDS[@]}"; do
        ckpt="checkpoints/cw_${method}_cw_subset_s${seed}.pt"
        log="checkpoints/cpu_${method}_s${seed}.log"
        echo "  ${method} seed=${seed}"
        python rl/continual_world.py \
            --method "$method" \
            --steps_per_task 200000 \
            --seed "$seed" --tasks cw_subset \
            --ckpt "$ckpt" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done
done

echo "${#PIDS[@]} runs launched"
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done at $(date)"
'
