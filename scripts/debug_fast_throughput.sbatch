#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -J fast_dbg
#SBATCH -o scripts/fast_debug.%j.out
#SBATCH -e scripts/fast_debug.%j.err

# Quick throughput test: fast SAC on window-close for 50K steps.
# Should show ~500+ sps vs ~80 sps in old version.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1

echo "Fast SAC throughput test at $(date)"
python rl/fast_cl_experiment.py \
    --method finetune \
    --tasks window-close \
    --steps_per_task 50000 \
    --n_envs 1024 \
    --grad_steps 32 \
    --seed 42 \
    --out fast_debug.json

echo "Done at $(date)"
'
