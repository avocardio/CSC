#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J sac_debug
#SBATCH -o scripts/sac_debug.%j.out
#SBATCH -e scripts/sac_debug.%j.err

# Quick test: can SAC learn CW tasks with proper hyperparams?
# Runs 4 tasks in parallel (one per GPU), 100K steps each.
# If any task reaches >0% success, the setup is correct.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p results

echo "SAC learning debug at $(date)"
echo "Testing: can SAC learn each CW task individually?"
echo "Config: n_envs=256, UTD=1.0, lr=1e-3, batch=128, 100K steps"

# Test 4 tasks in parallel — single task per GPU, no CL
for i in 0 1 2 3; do
    task=$(echo "window-close faucet-close handle-press-side peg-unplug-side" | cut -d" " -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$i python rl/cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 100000 --n_envs 256 --seed 42 \
        --out results/debug_${task}_s42.json \
        > results/debug_${task}_s42.log 2>&1 &
done
wait

echo ""
echo "=== SINGLE-TASK LEARNING RESULTS ==="
for task in window-close faucet-close handle-press-side peg-unplug-side; do
    echo -n "$task: "
    grep "avg_performance\|succ_curr" results/debug_${task}_s42.log 2>/dev/null | tail -3
    echo
done
echo "Done at $(date)"
'
