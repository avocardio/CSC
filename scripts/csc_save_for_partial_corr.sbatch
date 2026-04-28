#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 02:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J csc_save
#SBATCH -o scripts/csc_save.%j.out
#SBATCH -e scripts/csc_save.%j.err

# Train CSC on R18/CIFAR-100 t10 (3 seeds), save final model state for
# partial correlation rho(b, F | |w|) analysis.

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
    log="checkpoints/sup_cifar100_csc_t10_s${seed}_corr.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method csc --model resnet18 \
        --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
        --gamma_comp 0.001 --soft_beta 1.0 \
        --replay_per_task 200 --replay_ratio 0.5 --der_alpha 0.5 \
        --tag _corr --save_model_final \
        --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
