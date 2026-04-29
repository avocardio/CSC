#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 04:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J l2p_cifar
#SBATCH -o scripts/l2p_cifar.%j.out
#SBATCH -e scripts/l2p_cifar.%j.err

# P6.1 — L2P baseline on Split-CIFAR-100, 10 tasks, 3 seeds.
# Online-LoRA paper uses LR=0.003, prompt_pool=10, prompt_len=10, top_k=4.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1
git pull --ff-only origin main 2>/dev/null || true
mkdir -p output_l2p
cd vendor/l2p
DATA_DIR=/capstor/scratch/cscs/mkalcher/datasets

PIDS=()
i=0
for seed in 42 123 456; do
    out_dir="../../output_l2p/cifar100_s${seed}"
    log="../../output_l2p/cifar100_s${seed}.log"
    mkdir -p "$out_dir"
    gpu=$((i % 4))
    env -u SLURM_PROCID -u SLURM_NPROCS -u SLURM_NTASKS -u SLURM_TASKS_PER_NODE \
        -u RANK -u WORLD_SIZE -u LOCAL_RANK \
        CUDA_VISIBLE_DEVICES=$gpu PYTHONPATH=. \
        python -u main.py cifar100_l2p \
        --data-path "$DATA_DIR" --seed $seed \
        --output_dir "$out_dir" \
        > "$log" 2>&1 &
    PIDS+=($!)
    i=$((i+1))
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
