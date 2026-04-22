#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J single_all
#SBATCH -o scripts/single_all.%j.out
#SBATCH -e scripts/single_all.%j.err

# Single-task diagnostic: train SAC on each of the 10 CW tasks independently.
# 250K steps each. Tests whether our GPU env port learns each task.
# 4 GPUs, ~3 tasks per GPU sequential (~50 min each).

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "Single-task diagnostic for all 10 CW tasks at $(date)"

# GPU 0: hammer, push-wall, stick-pull
(for task in hammer push-wall stick-pull; do
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out single_${task}.json \
        > checkpoints/single_${task}.log 2>&1
done) &

# GPU 1: faucet-close, push-back, push
(for task in faucet-close push-back push; do
    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out single_${task}.json \
        > checkpoints/single_${task}.log 2>&1
done) &

# GPU 2: handle-press-side, shelf-place, peg-unplug-side
(for task in handle-press-side shelf-place peg-unplug-side; do
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out single_${task}.json \
        > checkpoints/single_${task}.log 2>&1
done) &

# GPU 3: window-close (control — should work)
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --seed 42 --out single_window-close.json \
    > checkpoints/single_window-close.log 2>&1 &

wait

echo ""
echo "=========================================="
echo "SINGLE-TASK RESULTS (250K steps each)"
echo "=========================================="
for task in hammer push-wall faucet-close push-back stick-pull handle-press-side push shelf-place window-close peg-unplug-side; do
    succ=$(grep "avg_performance" checkpoints/single_${task}.log 2>/dev/null | tail -1 | awk "{print \$2}")
    peak=$(grep "succ_curr" checkpoints/single_${task}.log 2>/dev/null | awk -F"succ_curr=" "{print \$2}" | awk "{print \$1}" | sort -n | tail -1)
    echo "  ${task}: eval=${succ:-FAIL}  peak_train=${peak:-0}"
done
echo "Done at $(date)"
'
