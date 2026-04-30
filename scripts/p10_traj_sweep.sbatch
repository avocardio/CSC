#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J p10_traj
#SBATCH -o scripts/p10_traj.%j.out
#SBATCH -e scripts/p10_traj.%j.err

# P10: bit-depth trajectory experiment.
# Tests whether channels SPECIALIZE on specific tasks and get REUSED later.
# Sweep gamma_comp ∈ {1e-3, 1e-2, 5e-2} to find a regime where bits move
# enough for differential specialization to be visible.
# 3 gamma × 3 seeds = 9 runs.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1
git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

PIDS=()
i=0
for gamma in 0.001 0.01 0.05; do
    for seed in 42 123 456; do
        gpu=$((i % 4))
        tag="_trajg${gamma}"
        log="checkpoints/sup_cifar100_csc_t10_s${seed}${tag}.log"
        CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
            --dataset cifar100 --method csc --model resnet18 \
            --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
            --gamma_comp "$gamma" --soft_beta 1.0 \
            --replay_per_task 200 --replay_ratio 0.5 --der_alpha 0.5 \
            --tag "$tag" \
            --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
            > "$log" 2>&1 &
        PIDS+=($!)
        i=$((i+1))
    done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
