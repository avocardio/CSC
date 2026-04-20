#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
#SBATCH -J cw10_smk
#SBATCH -o scripts/cw10_smoke.%j.out
#SBATCH -e scripts/cw10_smoke.%j.err

# Quick smoke test: all 5 methods on CW10 with only 10K steps/task.
# Verifies: no crashes, all tasks load, all methods run, eval works.
# Should finish in ~15 min.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "CW10 smoke test at $(date)"

# GPU 0: finetune + csc (sequential, both fast at 10K)
(CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks cw10 \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out smoke_finetune.json \
    > checkpoints/smoke_finetune.log 2>&1 && \
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method csc --tasks cw10 \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out smoke_csc.json \
    > checkpoints/smoke_csc.log 2>&1) &

# GPU 1: ewc
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method ewc --tasks cw10 \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out smoke_ewc.json \
    > checkpoints/smoke_ewc.log 2>&1 &

# GPU 2: replay
CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
    --method replay --tasks cw10 \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out smoke_replay.json \
    > checkpoints/smoke_replay.log 2>&1 &

# GPU 3: packnet
CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
    --method packnet --tasks cw10 \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out smoke_packnet.json \
    > checkpoints/smoke_packnet.log 2>&1 &

wait

echo ""
echo "=========================================="
echo "SMOKE TEST RESULTS"
echo "=========================================="
ALL_OK=true
for method in finetune ewc replay packnet csc; do
    echo "--- $method ---"
    if grep -q "DONE" checkpoints/smoke_${method}.log 2>/dev/null; then
        grep "DONE\|avg_performance\|avg_forgetting" checkpoints/smoke_${method}.log | tail -4
    else
        echo "FAILED or incomplete!"
        tail -5 checkpoints/smoke_${method}.log 2>/dev/null
        ALL_OK=false
    fi
    echo
done

if $ALL_OK; then
    echo "ALL METHODS PASSED SMOKE TEST"
else
    echo "SOME METHODS FAILED - check logs"
fi
echo "Done at $(date)"
'
