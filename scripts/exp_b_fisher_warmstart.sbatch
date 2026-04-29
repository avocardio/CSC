#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J exp_b_fish
#SBATCH -o scripts/exp_b_fish.%j.out
#SBATCH -e scripts/exp_b_fish.%j.err

# Experiment B: Fisher warm-start with bit-depth.
# Sweep K (n_batches in EWC's Fisher pass) for plain EWC and for bd-warmstart EWC.
# K in {1, 2, 5, 10, 20}. 3 seeds each. 2 methods. R18 / CIFAR-100 / 10 tasks.
# 5 K-values × 2 methods × 3 seeds = 30 runs. Round-robin on 4 GPUs.

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
for K in 1 2 5 10 20; do
  for method in ewc csc_bd_ewc_warm; do
    for seed in 42 123 456; do
      gpu=$((i % 4))
      tag="_K${K}"
      log="checkpoints/sup_cifar100_${method}_t10_s${seed}${tag}.log"
      CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
          --dataset cifar100 --method "$method" --model resnet18 \
          --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
          --gamma_comp 0.001 --soft_beta 1.0 --ewc_lambda 1000 \
          --ewc_n_batches "$K" --bd_init_scale 1e-3 \
          --replay_per_task 0 --der_alpha 0 \
          --tag "$tag" \
          --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
          > "$log" 2>&1 &
      PIDS+=($!)
      i=$((i+1))
    done
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
