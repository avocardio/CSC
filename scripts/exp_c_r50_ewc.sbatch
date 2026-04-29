#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J exp_c_r50
#SBATCH -o scripts/exp_c_r50.%j.out
#SBATCH -e scripts/exp_c_r50.%j.err

# Experiment C: bd-as-Fisher quality at larger model.
# Run EWC, csc_ewc, csc_bd_ewc on R50 (3 seeds each, 9 total runs).
# If bd-as-Fisher's gap to real Fisher narrows at R50 vs R18 (47/72=0.65),
# that's a "compute savings at scale" story.

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
for method in ewc csc_ewc csc_bd_ewc; do
  for seed in 42 123 456; do
    gpu=$((i % 4))
    log="checkpoints/sup_cifar100_${method}_resnet50_t10_s${seed}.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method "$method" --model resnet50 \
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
