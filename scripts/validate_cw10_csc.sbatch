#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -J cw10_csc
#SBATCH -o scripts/cw10_csc.%j.out
#SBATCH -e scripts/cw10_csc.%j.err

# Validation batch 2: CSC only (chained after batch 1)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

echo "CW10 validation CSC at $(date)"

python rl/cl_experiment.py \
    --method csc --tasks cw10 \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --seed 42 --out val_csc_s42.json \
    > checkpoints/val_csc_s42.log 2>&1

echo ""
echo "=== CSC RESULTS ==="
grep "avg_performance\|avg_forgetting\|backward_transfer\|forward_transfer\|DONE" \
    checkpoints/val_csc_s42.log 2>/dev/null | tail -6
echo "Done at $(date)"
'
