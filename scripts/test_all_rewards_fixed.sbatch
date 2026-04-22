#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J reward_fix
#SBATCH -o scripts/reward_fix.%j.out
#SBATCH -e scripts/reward_fix.%j.err

# Single-task 250K on all 10 tasks with fixed rewards + grasping physics.
# GPU 0: hammer, push-wall, stick-pull (3 sequential, ~2.5h total)
# GPU 1: faucet-close, push-back, push (3 sequential)
# GPU 2: handle-press-side, shelf-place, peg-unplug-side (3 sequential)
# GPU 3: window-close (control, then done)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "Full reward fix test at $(date)"

# GPU 0
(for task in hammer push-wall stick-pull; do
    echo "GPU0: $task"
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out rwfix_${task}.json \
        > checkpoints/rwfix_${task}.log 2>&1
done) &

# GPU 1
(for task in faucet-close push-back push; do
    echo "GPU1: $task"
    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out rwfix_${task}.json \
        > checkpoints/rwfix_${task}.log 2>&1
done) &

# GPU 2
(for task in handle-press-side shelf-place peg-unplug-side; do
    echo "GPU2: $task"
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out rwfix_${task}.json \
        > checkpoints/rwfix_${task}.log 2>&1
done) &

# GPU 3
(for task in window-close; do
    echo "GPU3: $task"
    CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 256 --batch_size 128 \
        --seed 42 --out rwfix_${task}.json \
        > checkpoints/rwfix_${task}.log 2>&1
done) &

wait

echo ""
echo "=========================================="
echo "REWARD FIX RESULTS (250K steps each)"
echo "=========================================="
for task in hammer push-wall faucet-close push-back stick-pull handle-press-side push shelf-place window-close peg-unplug-side; do
    eval=$(grep "avg_performance" checkpoints/rwfix_${task}.log 2>/dev/null | tail -1 | awk "{print \$2}")
    peak=$(grep "succ_curr" checkpoints/rwfix_${task}.log 2>/dev/null | awk -F"succ_curr=" "{print \$2}" | awk "{print \$1}" | sort -rn | head -1)
    echo "  ${task}: eval=${eval:-FAIL}  peak=${peak:-0}"
done
echo "Done at $(date)"
'
