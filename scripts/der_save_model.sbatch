#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 03:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J der_save
#SBATCH -o scripts/der_save.%j.out
#SBATCH -e scripts/der_save.%j.err

# Train DER++ on R18/CIFAR-100 t10 (3 seeds), save final model state for
# post-hoc quantization (PTQ baseline analysis).

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
    log="checkpoints/sup_cifar100_der_t10_s${seed}_save.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method der --model resnet18 \
        --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
        --replay_per_task 200 --replay_ratio 0.5 \
        --der_alpha 0.5 --der_beta 1.0 \
        --tag _save --save_model_final \
        --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
