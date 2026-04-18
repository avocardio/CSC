#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J fast_aclp
#SBATCH -o scripts/fast_aclip.%j.out
#SBATCH -e scripts/fast_aclip.%j.err

# Test alpha-clipped fast SAC (UTD=0.2, batch=512) on 4 CW tasks.
# Alpha clamped to [0.007, 0.2] to prevent explosion/collapse.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p results

git pull --ff-only origin main 2>/dev/null || true

echo "Alpha-clipped fast SAC test at $(date)"

for i in 0 1 2 3; do
    task=$(echo "window-close faucet-close handle-press-side peg-unplug-side" | cut -d" " -f$((i+1)))
    CUDA_VISIBLE_DEVICES=$i python rl/fast_cl_experiment.py \
        --method finetune --tasks $task \
        --steps_per_task 250000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
        --seed 42 --out fast_aclip_${task}.json \
        > results/fast_aclip_${task}.log 2>&1 &
done
wait

echo ""
echo "=== ALPHA-CLIPPED RESULTS ==="
for task in window-close faucet-close handle-press-side peg-unplug-side; do
    echo "--- $task ---"
    grep "sps=\|succ=\|DONE\|avg_" results/fast_aclip_${task}.log 2>/dev/null | tail -4
    echo
done
echo "Done at $(date)"
'
