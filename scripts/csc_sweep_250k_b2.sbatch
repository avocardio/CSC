#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J csc_swp2
#SBATCH -o scripts/csc_sweep2.%j.out
#SBATCH -e scripts/csc_sweep2.%j.err

# CSC hyperparameter sweep batch 2 (8 configs: beta=2.0 and beta=5.0)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "CSC sweep batch 2 at $(date)"

# GPU 0: gc=0.001 beta=2.0, then gc=0.001 beta=5.0
(CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.001 --grad_scale_beta 2.0 \
    --seed 42 --run_name csc_gc001_b20 --out csc_gc001_b20.json \
    > checkpoints/csc_gc001_b20.log 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.001 --grad_scale_beta 5.0 \
    --seed 42 --run_name csc_gc001_b50 --out csc_gc001_b50.json \
    > checkpoints/csc_gc001_b50.log 2>&1) &

# GPU 1: gc=0.01 beta=2.0, then gc=0.01 beta=5.0
(CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.01 --grad_scale_beta 2.0 \
    --seed 42 --run_name csc_gc01_b20 --out csc_gc01_b20.json \
    > checkpoints/csc_gc01_b20.log 2>&1 && \
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.01 --grad_scale_beta 5.0 \
    --seed 42 --run_name csc_gc01_b50 --out csc_gc01_b50.json \
    > checkpoints/csc_gc01_b50.log 2>&1) &

# GPU 2: gc=0.05 beta=2.0, then gc=0.05 beta=5.0
(CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.05 --grad_scale_beta 2.0 \
    --seed 42 --run_name csc_gc05_b20 --out csc_gc05_b20.json \
    > checkpoints/csc_gc05_b20.log 2>&1 && \
CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.05 --grad_scale_beta 5.0 \
    --seed 42 --run_name csc_gc05_b50 --out csc_gc05_b50.json \
    > checkpoints/csc_gc05_b50.log 2>&1) &

# GPU 3: gc=0.1 beta=2.0, then gc=0.1 beta=5.0
(CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.1 --grad_scale_beta 2.0 \
    --seed 42 --run_name csc_gc10_b20 --out csc_gc10_b20.json \
    > checkpoints/csc_gc10_b20.log 2>&1 && \
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method csc --tasks cw_learnable \
    --steps_per_task 250000 --n_envs 256 --batch_size 128 \
    --gamma_comp 0.1 --grad_scale_beta 5.0 \
    --seed 42 --run_name csc_gc10_b50 --out csc_gc10_b50.json \
    > checkpoints/csc_gc10_b50.log 2>&1) &

wait
echo "Batch 2 done at $(date)"

echo ""
echo "=== CSC SWEEP BATCH 2 RESULTS ==="
for name in gc001_b20 gc001_b50 gc01_b20 gc01_b50 gc05_b20 gc05_b50 gc10_b20 gc10_b50; do
    echo "--- csc_${name} ---"
    grep "avg_performance\|avg_forgetting" checkpoints/csc_${name}.log 2>/dev/null | tail -2
    echo
done
'
