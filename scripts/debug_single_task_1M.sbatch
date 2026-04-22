#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J single_1M
#SBATCH -o scripts/single_1M.%j.out
#SBATCH -e scripts/single_1M.%j.err

# Single-task at 1M steps for the 6 tasks that need more budget.
# ~3.5h per task at 80 sps. 2 tasks per GPU = ~7h per GPU.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "1M step single-task for hard tasks at $(date)"

# GPU 0: push-wall, push
(for task in push-wall push; do
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 1000000 --n_envs 256 --batch_size 128 \
        --seed 42 --out single1M_${task}.json \
        > checkpoints/single1M_${task}.log 2>&1
done) &

# GPU 1: push-back, shelf-place
(for task in push-back shelf-place; do
    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 1000000 --n_envs 256 --batch_size 128 \
        --seed 42 --out single1M_${task}.json \
        > checkpoints/single1M_${task}.log 2>&1
done) &

# GPU 2: stick-pull, peg-unplug-side
(for task in stick-pull peg-unplug-side; do
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 1000000 --n_envs 256 --batch_size 128 \
        --seed 42 --out single1M_${task}.json \
        > checkpoints/single1M_${task}.log 2>&1
done) &

# GPU 3: hammer at 1M (verify it improves beyond 43.8%)
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method finetune --tasks hammer \
    --steps_per_task 1000000 --n_envs 256 --batch_size 128 \
    --seed 42 --out single1M_hammer.json \
    > checkpoints/single1M_hammer.log 2>&1 &

wait

echo ""
echo "=========================================="
echo "1M SINGLE-TASK RESULTS"
echo "=========================================="
for task in push-wall push push-back shelf-place stick-pull peg-unplug-side hammer; do
    eval=$(grep "avg_performance" checkpoints/single1M_${task}.log 2>/dev/null | tail -1 | awk "{print \$2}")
    peak=$(grep "succ_curr" checkpoints/single1M_${task}.log 2>/dev/null | awk -F"succ_curr=" "{print \$2}" | awk "{print \$1}" | sort -rn | head -1)
    echo "  ${task}: eval=${eval:-FAIL}  peak=${peak:-0}"
done
echo "Done at $(date)"
'
