#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J fix_alpha
#SBATCH -o scripts/fix_alpha.%j.out
#SBATCH -e scripts/fix_alpha.%j.err

# Test fixed alpha values vs auto-tuned alpha.
# GPU 0: fixed_alpha=0.01 (low exploration)
# GPU 1: fixed_alpha=0.05 (moderate)
# GPU 2: fixed_alpha=0.1  (standard)
# GPU 3: auto-alpha with clipping (baseline)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p results

git pull --ff-only origin main 2>/dev/null || true

echo "Fixed alpha sweep at $(date)"

# GPU 0: fixed_alpha=0.01
CUDA_VISIBLE_DEVICES=0 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 500000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
    --fixed_alpha 0.01 --seed 42 --out fix_a001.json \
    > results/fix_a001.log 2>&1 &

# GPU 1: fixed_alpha=0.05
CUDA_VISIBLE_DEVICES=1 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 500000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
    --fixed_alpha 0.05 --seed 42 --out fix_a005.json \
    > results/fix_a005.log 2>&1 &

# GPU 2: fixed_alpha=0.1
CUDA_VISIBLE_DEVICES=2 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 500000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
    --fixed_alpha 0.1 --seed 42 --out fix_a01.json \
    > results/fix_a01.log 2>&1 &

# GPU 3: auto-alpha (clipped)
CUDA_VISIBLE_DEVICES=3 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 500000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
    --seed 42 --out fix_auto.json \
    > results/fix_auto.log 2>&1 &

wait

echo ""
echo "=== FIXED ALPHA SWEEP RESULTS ==="
for name in a001 a005 a01 auto; do
    echo "--- $name ---"
    grep "sps=\|succ=\|DONE\|avg_" results/fix_${name}.log 2>/dev/null | tail -5
    echo
done
echo "Done at $(date)"
'
