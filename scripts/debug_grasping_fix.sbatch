#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J grasp_fix
#SBATCH -o scripts/grasp_fix.%j.out
#SBATCH -e scripts/grasp_fix.%j.err

# Test grasping tasks with restored geom_margin (non-zero contacts).
# Quick 50K step test on the 4 hardest grasping tasks.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "Grasping fix test at $(date)"

# GPU 0: hammer (50K steps)
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks hammer \
    --steps_per_task 50000 --n_envs 256 --batch_size 128 \
    --seed 42 --out grasp_hammer.json \
    > checkpoints/grasp_hammer.log 2>&1 &

# GPU 1: push (50K)
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method finetune --tasks push \
    --steps_per_task 50000 --n_envs 256 --batch_size 128 \
    --seed 42 --out grasp_push.json \
    > checkpoints/grasp_push.log 2>&1 &

# GPU 2: shelf-place (50K)
CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method finetune --tasks shelf-place \
    --steps_per_task 50000 --n_envs 256 --batch_size 128 \
    --seed 42 --out grasp_shelf.json \
    > checkpoints/grasp_shelf.log 2>&1 &

# GPU 3: stick-pull (50K)
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method finetune --tasks stick-pull \
    --steps_per_task 50000 --n_envs 256 --batch_size 128 \
    --seed 42 --out grasp_stick.json \
    > checkpoints/grasp_stick.log 2>&1 &

wait

echo ""
echo "=== GRASPING FIX RESULTS ==="
for task in hammer push shelf-place stick-pull; do
    echo "--- $task ---"
    peak=$(grep "succ_curr" checkpoints/grasp_${task}.log 2>/dev/null | awk -F"succ_curr=" "{print \$2}" | awk "{print \$1}" | sort -rn | head -1)
    eval=$(grep "avg_performance" checkpoints/grasp_${task}.log 2>/dev/null | tail -1)
    echo "  peak_train=$peak  $eval"
    echo
done
echo "Done at $(date)"
'
