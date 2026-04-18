#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J fast_500k
#SBATCH -o scripts/fast_500k.%j.out
#SBATCH -e scripts/fast_500k.%j.err

# Auto-alpha (clipped) at 500K steps on 4 learnable tasks.
# Validated: window-close reaches 100% eval at 500K with this config.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p results

git pull --ff-only origin main 2>/dev/null || true

echo "Fast 500K test on 4 tasks at $(date)"

for i in 0 1 2 3; do
    task=$(echo "window-close faucet-close handle-press-side peg-unplug-side" | cut -d" " -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$i python rl/fast_cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 500000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
        --seed 42 --out fast_500k_${task}.json \
        > results/fast_500k_${task}.log 2>&1 &
done
wait

echo ""
echo "=== 500K RESULTS ==="
for task in window-close faucet-close handle-press-side peg-unplug-side; do
    echo "--- $task ---"
    grep "sps=\|DONE\|avg_" results/fast_500k_${task}.log 2>/dev/null | tail -5
    echo
done
echo "Done at $(date)"
'
