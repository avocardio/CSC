#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:0
#SBATCH -J p4_bdmas
#SBATCH -o scripts/p4_bdmas.%j.out
#SBATCH -e scripts/p4_bdmas.%j.err
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
PIDS=()
for seed in 42 123 456 789 1234; do
    log="checkpoints/cpu_csc_bd_mas_s${seed}.log"
    ckpt="checkpoints/cw_csc_bd_mas_cw_subset_s${seed}.pt"
    rm -f "$log" "$ckpt"*
    python rl/continual_world.py \
        --method csc_bd_mas \
        --steps_per_task 200000 \
        --seed "$seed" --tasks cw_subset \
        --gamma_comp 0.001 --grad_scale_beta 1.0 \
        --cl_reg_coef 1e4 \
        --ckpt "$ckpt" \
        > "$log" 2>&1 &
    PIDS+=($!)
done
for pid in "${PIDS[@]}"; do wait "$pid" || echo "  pid $pid non-zero"; done
echo "Done"
'
