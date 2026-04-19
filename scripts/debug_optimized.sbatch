#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J opt_bench
#SBATCH -o scripts/opt_bench.%j.out
#SBATCH -e scripts/opt_bench.%j.err

# Benchmark: torch.compile + TF32 + fused Adam + set_to_none
# GPU 0: finetune, batch=128 (proven config, now optimized)
# GPU 1: finetune, batch=256 (larger batch test)
# GPU 2: csc, batch=128 (CSC with optimizations)
# GPU 3: ewc, batch=128 (EWC with optimizations)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "Optimized benchmark at $(date)"

# GPU 0: finetune batch=128 (baseline optimized)
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks window-close --batch_size 128 \
    --steps_per_task 50000 --n_envs 256 --seed 42 \
    --out opt_ft128.json \
    > checkpoints/opt_ft128.log 2>&1 &

# GPU 1: finetune batch=256
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method finetune --tasks window-close --batch_size 256 \
    --steps_per_task 50000 --n_envs 256 --seed 42 \
    --out opt_ft256.json \
    > checkpoints/opt_ft256.log 2>&1 &

# GPU 2: csc batch=128
CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method csc --tasks window-close --batch_size 128 \
    --steps_per_task 50000 --n_envs 256 --seed 42 \
    --out opt_csc128.json \
    > checkpoints/opt_csc128.log 2>&1 &

# GPU 3: ewc batch=128
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method ewc --tasks window-close --batch_size 128 \
    --steps_per_task 50000 --n_envs 256 --seed 42 \
    --out opt_ewc128.json \
    > checkpoints/opt_ewc128.log 2>&1 &

wait

echo ""
echo "=== BENCHMARK RESULTS (SPS comparison) ==="
echo "Previous baseline: ~80 sps"
echo ""
for name in ft128 ft256 csc128 ewc128; do
    echo "--- $name ---"
    grep "sps=" checkpoints/opt_${name}.log 2>/dev/null | tail -3
    echo
done
echo "Done at $(date)"
'
