#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:0
#SBATCH -J cpu_cw10
#SBATCH -o scripts/cpu_cw10.%j.out
#SBATCH -e scripts/cpu_cw10.%j.err

# CPU CW10 benchmark: all methods × all seeds in parallel on 288 ARM cores.
# Each experiment uses 1 core at ~100 sps → 2.7h/task, 27h/CW10.
# With checkpointing: first job does ~4 tasks, second job resumes.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1

git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

echo "CPU CW10 benchmark at $(date)"
echo "Methods: finetune, ewc, replay, csc"
echo "Seeds: 42, 123, 456"

METHODS=(finetune ewc replay csc)
SEEDS=(42 123 456)
PIDS=()

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "Starting: ${method} seed=${seed}"
        python rl/continual_world.py \
            --method $method \
            --steps_per_task 1000000 \
            --seed $seed \
            --utd 1.0 \
            --tag "_cpu" \
            > checkpoints/cpu_${method}_s${seed}.log 2>&1 &
        PIDS+=($!)
    done
done

echo "${#PIDS[@]} experiments launched on $(nproc) cores"
echo "Waiting..."
wait

echo ""
echo "=========================================="
echo "CPU CW10 BENCHMARK RESULTS"
echo "=========================================="
for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        echo "--- ${method} s${seed} ---"
        tail -12 checkpoints/cpu_${method}_s${seed}.log 2>/dev/null | head -12
        echo
    done
done
echo "Done at $(date)"
'
