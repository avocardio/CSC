#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J p5_1_ctrl
#SBATCH -o scripts/p5_1_ctrl.%j.out
#SBATCH -e scripts/p5_1_ctrl.%j.err

# P5.1 — control ablation. Compare bd vs random vs |w| vs Fisher as the
# importance signal in EWC's penalty. Same gamma_comp + soft_protect for all.
# 4 methods x 3 seeds = 12 runs.

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
# csc_ewc = bd + real Fisher (existing). Reference: ewc alone (Fisher only).
for method in ewc csc_random_ewc csc_magnitude_ewc csc_bd_ewc; do
  for seed in 42 123 456; do
    gpu=$((i % 4))
    log="checkpoints/sup_cifar100_${method}_t10_s${seed}.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method "$method" --model resnet18 \
        --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
        --gamma_comp 0.001 --soft_beta 1.0 --ewc_lambda 1000 \
        --replay_per_task 0 --der_alpha 0 \
        --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
