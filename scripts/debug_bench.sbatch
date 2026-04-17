#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -J csc_bench
#SBATCH -o scripts/debug_bench.%j.out
#SBATCH -e scripts/debug_bench.%j.err

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail

module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"

export PYTHONNOUSERSITE=1
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export PYTHONUNBUFFERED=1

export TORCH_HOME=/capstor/scratch/cscs/$USER/CSC/.cache/torch
export TRITON_CACHE_DIR=/capstor/scratch/cscs/$USER/CSC/.cache/triton
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR"

bash scripts/debug_bench.sh
echo "Done at $(date)"
'
