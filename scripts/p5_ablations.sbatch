#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J p5_abl
#SBATCH -o scripts/p5_abl.%j.out
#SBATCH -e scripts/p5_abl.%j.err

# P5.2 + P5.3 — gamma_comp sweep and bd_init_scale sweep on R18 / CIFAR-100.

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

# P5.2: gamma_comp sweep — see how bd-warm-start uplift changes with compression pressure.
for gamma in 0.0001 0.001 0.01 0.05; do
  for seed in 42 123; do
    gpu=$((i % 4))
    tag="_g${gamma}"
    log="checkpoints/sup_cifar100_csc_bd_ewc_warm_t10_s${seed}${tag}.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method csc_bd_ewc_warm --model resnet18 \
        --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
        --gamma_comp "$gamma" --soft_beta 1.0 --ewc_lambda 1000 \
        --ewc_n_batches 5 --bd_init_scale 1e-3 \
        --replay_per_task 0 --der_alpha 0 \
        --tag "$tag" \
        --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
  done
done

# P5.3: bd_init_scale sweep — find optimal coupling between bd and Fisher.
for scale in 0.0001 0.01 0.1 1.0; do
  for seed in 42 123; do
    gpu=$((i % 4))
    tag="_bs${scale}"
    log="checkpoints/sup_cifar100_csc_bd_ewc_warm_t10_s${seed}${tag}.log"
    CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
        --dataset cifar100 --method csc_bd_ewc_warm --model resnet18 \
        --num_tasks 10 --epochs_per_task 50 --seed "$seed" \
        --gamma_comp 0.001 --soft_beta 1.0 --ewc_lambda 1000 \
        --ewc_n_batches 5 --bd_init_scale "$scale" \
        --replay_per_task 0 --der_alpha 0 \
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
