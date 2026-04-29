#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J exp_d_olora
#SBATCH -o scripts/exp_d_olora.%j.out
#SBATCH -e scripts/exp_d_olora.%j.err

# Experiment D: Online-LoRA + CSC integration on Split-CIFAR-100, 10 tasks.
# Three configurations (3 seeds each):
#   olora_mas    : original Online-LoRA, MAS-style 4-sample omega.
#   olora_csc_mas: same but with CSC quantizer wrapping LoRA wnew layers
#                  (compression objective on, omega still from MAS gradients).
#                  Tests whether CSC wrap alone affects accuracy.
#   olora_csc_bd : CSC quantizer + bit-depth as omega (instead of MAS).
#                  The headline experiment.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1
pip install --quiet timm safetensors torch_optimizer 2>/dev/null || true

git pull --ff-only origin main 2>/dev/null || true
mkdir -p output_olora

cd vendor/online_lora

DATA_DIR=/capstor/scratch/cscs/mkalcher/datasets

PIDS=()
i=0
for cfg in olora_mas olora_csc_mas olora_csc_bd; do
  for seed in 42 123 456; do
    gpu=$((i % 4))
    case $cfg in
      olora_mas)     extra="" ;;
      olora_csc_mas) extra="--csc --gamma_comp 0.001" ;;
      olora_csc_bd)  extra="--csc --gamma_comp 0.001 --importance bd" ;;
    esac
    out_dir="../../output_olora/${cfg}_s${seed}"
    log="../../output_olora/${cfg}_s${seed}.log"
    mkdir -p "$out_dir"
    echo "[GPU $gpu] $cfg seed=$seed -> $out_dir"
    # Unset SLURM_* so init_distributed_mode falls through to "not distributed"
    # (we manage parallelism ourselves via 4 GPUs in one node).
    env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_NTASKS -u SLURM_TASKS_PER_NODE \
        -u RANK -u WORLD_SIZE -u LOCAL_RANK \
        CUDA_VISIBLE_DEVICES=$gpu \
        PYTHONPATH=. \
        python -u main.py \
        cifar100_online_lora \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --MAS-weight 2000 \
        --lr 0.0002 \
        --num_tasks 10 \
        --loss-window-variance-threshold 0.1 \
        --loss-window-mean-threshold 6.0 \
        --data-path "$DATA_DIR" \
        --seed $seed \
        --output_dir "$out_dir" \
        $extra \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
  done
done

for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
