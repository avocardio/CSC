#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J fast_swp
#SBATCH -o scripts/fast_sweep.%j.out
#SBATCH -e scripts/fast_sweep.%j.err

# Sweep UTD/batch to find the sweet spot between speed and learning.
# GPU 0: UTD=0.1, batch=1024 (102 grad steps per 1024 envs)
# GPU 1: UTD=0.2, batch=512  (205 grad steps per 1024 envs)
# GPU 2: UTD=0.5, batch=256  (512 grad steps per 1024 envs)
# GPU 3: UTD=0.03, batch=4096 (32 grad steps — baseline fast)

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p results

echo "UTD/batch sweep on window-close at $(date)"

# All use n_envs=1024, 250K steps, window-close only
# GPU 0: UTD=0.1, batch=1024
CUDA_VISIBLE_DEVICES=0 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 250000 --n_envs 1024 --grad_steps 102 --batch_size 1024 \
    --seed 42 --out fast_utd01.json \
    > results/fast_utd01.log 2>&1 &

# GPU 1: UTD=0.2, batch=512
CUDA_VISIBLE_DEVICES=1 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 250000 --n_envs 1024 --grad_steps 205 --batch_size 512 \
    --seed 42 --out fast_utd02.json \
    > results/fast_utd02.log 2>&1 &

# GPU 2: UTD=0.5, batch=256
CUDA_VISIBLE_DEVICES=2 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 250000 --n_envs 1024 --grad_steps 512 --batch_size 256 \
    --seed 42 --out fast_utd05.json \
    > results/fast_utd05.log 2>&1 &

# GPU 3: UTD=0.03, batch=4096 (fast baseline)
CUDA_VISIBLE_DEVICES=3 python rl/fast_cl_experiment.py \
    --method finetune --tasks window-close \
    --steps_per_task 250000 --n_envs 1024 --grad_steps 32 --batch_size 4096 \
    --seed 42 --out fast_utd003.json \
    > results/fast_utd003.log 2>&1 &

wait

echo ""
echo "=== UTD SWEEP RESULTS ==="
for name in utd003 utd01 utd02 utd05; do
    echo "--- $name ---"
    grep "sps=\\|succ=\\|DONE\\|avg_" results/fast_${name}.log 2>/dev/null | tail -4
    echo
done
echo "Done at $(date)"
'
