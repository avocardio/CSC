#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J ewc_base
#SBATCH -o scripts/ewc_base.%j.out
#SBATCH -e scripts/ewc_base.%j.err

# CIFAR-100 R18 EWC baseline (no replay/DER), 3 seeds, for the csc_ewc comparison.
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
for seed in 42 123 456; do
    gpu=$((i % 4))
    log="checkpoints/sup_cifar100_ewc_t10_s${seed}.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method ewc --model resnet18 \
        --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
        --ewc_lambda 1000 --replay_per_task 0 --der_alpha 0 \
        --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
'
