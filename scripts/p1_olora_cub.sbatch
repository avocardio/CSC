#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J olora_cub
#SBATCH -o scripts/olora_cub.%j.out
#SBATCH -e scripts/olora_cub.%j.err

# P1.4 + P2.4 — Online-LoRA reproduction on CUB-200 + CSC.
# Published HPs (appendix B): mean=24.0, var=1.0, lambda=2000, num_tasks=10.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1
git pull --ff-only origin main 2>/dev/null || true
mkdir -p output_olora_cub
cd vendor/online_lora
DATA_DIR=/capstor/scratch/cscs/mkalcher/datasets
PIDS=()
i=0
for cfg in olora_mas olora_csc_mas olora_csc_bd; do
  for seed in 42 123 456; do
    case $cfg in
      olora_mas)     extra="" ;;
      olora_csc_mas) extra="--csc --gamma_comp 0.001" ;;
      olora_csc_bd)  extra="--csc --gamma_comp 0.001 --importance bd" ;;
    esac
    out_dir="../../output_olora_cub/${cfg}_s${seed}"
    log="../../output_olora_cub/${cfg}_s${seed}.log"
    mkdir -p "$out_dir"
    gpu=$((i % 4))
    env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_NTASKS -u SLURM_TASKS_PER_NODE \
        -u RANK -u WORLD_SIZE -u LOCAL_RANK \
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
        python -u main.py cub200_online_lora \
        --model vit_base_patch16_224 \
        --batch-size 64 --MAS-weight 2000 --lr 0.0002 --num_tasks 10 \
        --loss-window-mean-threshold 24.0 --loss-window-variance-threshold 1.0 \
        --data-path "$DATA_DIR" --seed $seed \
        --output_dir "$out_dir" \
        $extra > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
