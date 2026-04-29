#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J p3_pmnist
#SBATCH -o scripts/p3_pmnist.%j.out
#SBATCH -e scripts/p3_pmnist.%j.err

# P3.4: bd warm-start on PMNIST MLP (10 tasks; 50 tasks if compute permits).

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
for nt in 10 50; do
  for cfg in "ewc 20" "csc_bd_ewc_warm 1" "csc_bd_ewc_warm 5" "csc_bd_ewc_warm 20"; do
    method=${cfg%% *}
    K=${cfg##* }
    for seed in 42 123 456; do
      gpu=$((i % 4))
      tag="_K${K}"
      log="checkpoints/sup_pmnist_${method}_t${nt}_s${seed}${tag}.log"
      CUDA_VISIBLE_DEVICES=$gpu python -u experiments/run_supervised.py \
          --dataset pmnist --method "$method" --model mlp \
          --num_tasks "$nt" --epochs_per_task 10 --seed "$seed" \
          --gamma_comp 0.001 --soft_beta 1.0 --ewc_lambda 1000 \
          --ewc_n_batches "$K" --bd_init_scale 1e-3 \
          --replay_per_task 0 --der_alpha 0 \
          --tag "$tag" \
          --data_root /capstor/scratch/cscs/mkalcher/datasets/mnist \
          > "$log" 2>&1 &
      PIDS+=($!)
      i=$((i+1))
    done
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
