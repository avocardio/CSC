#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:0
#SBATCH -J csc_clonex
#SBATCH -o scripts/csc_clonex.%j.out
#SBATCH -e scripts/csc_clonex.%j.err

# Re-run csc_clonex 5 seeds from scratch with longer wall.
# Previous run hit wall at task 3 step 100K because BC snapshots aren't
# in save_checkpoint, so resume would skip BC distillation.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -uo pipefail
module load python/3.12.5
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
export CUDA_VISIBLE_DEVICES=""

git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

# Wipe stale csc_clonex artifacts so we re-run cleanly.
rm -f checkpoints/cw_csc_clonex_cw_subset_s*.pt
rm -f checkpoints/cw_csc_clonex_cw_subset_s*_final.json
rm -f checkpoints/cpu_csc_clonex_s*.log

echo "csc_clonex sweep at $(date)"

SEEDS=(42 123 456 789 1234)
PIDS=()
for seed in "${SEEDS[@]}"; do
    ckpt="checkpoints/cw_csc_clonex_cw_subset_s${seed}.pt"
    log="checkpoints/cpu_csc_clonex_s${seed}.log"
    echo "  csc_clonex seed=${seed}"
    python rl/continual_world.py \
        --method csc_clonex \
        --steps_per_task 200000 \
        --seed "$seed" --tasks cw_subset \
        --ckpt "$ckpt" \
        > "$log" 2>&1 &
    PIDS+=($!)
done

echo "${#PIDS[@]} runs launched"
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done at $(date)"
'
