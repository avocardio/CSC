#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:0
#SBATCH -J rl_redo
#SBATCH -o scripts/rl_redo.%j.out
#SBATCH -e scripts/rl_redo.%j.err

# Re-run replay + csc seeds for CW subset with the alpha-explosion fix
# (commit 881c2e3 — alpha update only on current-task slice).
# Other methods from job 1955615 (finetune/l2/ewc/mas) are valid as-is.

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

# Wipe stale logs / mid-task ckpts so we rerun cleanly under the alpha fix.
rm -f checkpoints/cpu_replay_s*.log checkpoints/cw_replay_cw_subset_s*.pt
rm -f checkpoints/cpu_csc_s*.log checkpoints/cw_csc_cw_subset_s*.pt
rm -f checkpoints/cw_replay_cw_subset_s*_final.json
rm -f checkpoints/cw_csc_cw_subset_s*_final.json

echo "RL replay+csc redo at $(date)"

METHODS=(replay csc)
SEEDS=(42 123 456 789 1234)
PIDS=()

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        ckpt="checkpoints/cw_${method}_cw_subset_s${seed}.pt"
        log="checkpoints/cpu_${method}_s${seed}.log"
        echo "Launching: ${method} seed=${seed}"
        python rl/continual_world.py \
            --method "$method" \
            --steps_per_task 200000 \
            --seed "$seed" \
            --tasks cw_subset \
            --ckpt "$ckpt" \
            > "$log" 2>&1 &
        PIDS+=($!)
    done
done

echo "${#PIDS[@]} jobs launched"
for pid in "${PIDS[@]}"; do
    wait $pid || echo "  pid $pid exited non-zero"
done

echo "Done at $(date)"
'
