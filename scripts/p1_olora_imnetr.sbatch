#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J olora_imnetr
#SBATCH -o scripts/olora_imnetr.%j.out
#SBATCH -e scripts/olora_imnetr.%j.err

# P1.2 + P2.2 — Online-LoRA reproduction on Split-ImageNet-R + CSC ablation.
# Published HPs (appendix B): mean=5.2, var=0.02, lambda=2000.
# Three configs (mas baseline, csc+mas control, csc+bd) x 3 seeds = 9 runs.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1
git pull --ff-only origin main 2>/dev/null || true
mkdir -p output_olora_imnetr
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
    out_dir="../../output_olora_imnetr/${cfg}_s${seed}"
    log="../../output_olora_imnetr/${cfg}_s${seed}.log"
    mkdir -p "$out_dir"
    gpu=$((i % 4))
    env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_NTASKS -u SLURM_TASKS_PER_NODE \
        -u RANK -u WORLD_SIZE -u LOCAL_RANK \
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
        python -u main.py imagenetR_online_lora \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --MAS-weight 2000 \
        --lr 0.0002 \
        --num_tasks 10 \
        --loss-window-mean-threshold 5.2 \
        --loss-window-variance-threshold 0.02 \
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
