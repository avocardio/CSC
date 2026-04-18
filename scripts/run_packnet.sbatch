#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=3
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:3
#SBATCH -J csc_pn
#SBATCH -o scripts/packnet.%j.out
#SBATCH -e scripts/packnet.%j.err

# PackNet baseline: 3 seeds on 3 GPUs (one seed per GPU)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
export TORCH_HOME=/capstor/scratch/cscs/$USER/CSC/.cache/torch
export TRITON_CACHE_DIR=/capstor/scratch/cscs/$USER/CSC/.cache/triton
mkdir -p "$TORCH_HOME" "$TRITON_CACHE_DIR" results

TASKS=cw_learnable
STEPS=250000
ENVS=256

echo "PackNet experiments at $(date)"

CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method packnet --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed 42 \
    --out results/cw_packnet_s42.json \
    > results/cw_packnet_s42.log 2>&1 &

CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method packnet --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed 1 \
    --out results/cw_packnet_s1.json \
    > results/cw_packnet_s1.log 2>&1 &

CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method packnet --tasks $TASKS --steps_per_task $STEPS \
    --n_envs $ENVS --seed 2 \
    --out results/cw_packnet_s2.json \
    > results/cw_packnet_s2.log 2>&1 &

wait

echo ""
echo "=== PACKNET RESULTS ==="
for s in 42 1 2; do
    echo -n "  s$s: "
    grep "avg_performance\|avg_forgetting" results/cw_packnet_s${s}.log 2>/dev/null | tr "\n" " " || echo "missing"
    echo
done
echo "Done at $(date)"
'
