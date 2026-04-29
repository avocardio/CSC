#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J p3_warm
#SBATCH -o scripts/p3_warm.%j.out
#SBATCH -e scripts/p3_warm.%j.err

# P3.1 P3.2 P3.3: bd warm-start (csc_bd_ewc_warm) on R50, R101, ConvNeXt-T.
# K-sweep {1, 5, 20} (cheaper than the R18 5-K version, just confirms shape).
# Plus plain ewc baselines at K=20 for the comparison.
# Total: 3 archs × 4 configs × 3 seeds = 36 runs, ~6h on 4 GPUs.

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
for arch in resnet50 resnet101 convnext_tiny; do
  for cfg in "ewc 20" "csc_bd_ewc_warm 1" "csc_bd_ewc_warm 5" "csc_bd_ewc_warm 20"; do
    method=${cfg%% *}
    K=${cfg##* }
    for seed in 42 123 456; do
      gpu=$((i % 4))
      tag="_K${K}"
      log="checkpoints/sup_cifar100_${method}_${arch}_t10_s${seed}${tag}.log"
      CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
          --dataset cifar100 --method "$method" --model "$arch" \
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
