#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J olora_repro
#SBATCH -o scripts/olora_repro.%j.out
#SBATCH -e scripts/olora_repro.%j.err

# P1.1 Online-LoRA reproduction sweep on CIFAR-100.
# Goal: identify the (LR, mean_threshold, variance_threshold) combo that
# matches their published ~60% accuracy. Runs UNMODIFIED upstream code.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1
pip install --quiet timm safetensors torch_optimizer 2>/dev/null || true

git pull --ff-only origin main 2>/dev/null || true
mkdir -p output_olora_repro
cd vendor/online_lora

DATA_DIR=/capstor/scratch/cscs/mkalcher/datasets

# Published Online-LoRA appendix B Table 8: CIFAR-100 mean=2.6, var=0.03.
# Sweep around that with LR ∈ {1e-4, 2e-4, 3e-4} × thresholds = 9 runs total.
PIDS=()
i=0
for lr in 0.0001 0.0002 0.0003; do
  for thr in "2.6_0.03" "2.2_0.02" "3.0_0.06"; do
    mean=${thr%_*}
    var=${thr#*_}
    name="lr${lr}_m${mean}_v${var}"
    out_dir="../../output_olora_repro/${name}"
    log="../../output_olora_repro/${name}.log"
    mkdir -p "$out_dir"
    gpu=$((i % 4))
    echo "[GPU $gpu] $name"
    env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_NTASKS -u SLURM_TASKS_PER_NODE \
        -u RANK -u WORLD_SIZE -u LOCAL_RANK \
        CUDA_VISIBLE_DEVICES=$gpu \
        PYTHONPATH=. \
        python -u main.py \
        cifar100_online_lora \
        --model vit_base_patch16_224 \
        --batch-size 64 \
        --MAS-weight 2000 \
        --lr "$lr" \
        --num_tasks 10 \
        --loss-window-mean-threshold "$mean" \
        --loss-window-variance-threshold "$var" \
        --data-path "$DATA_DIR" \
        --seed 42 \
        --output_dir "$out_dir" \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
  done
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
