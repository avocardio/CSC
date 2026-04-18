#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 06:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=4
#SBATCH --cpus-per-task=64
#SBATCH --gres=gpu:4
#SBATCH -J csc_mega
#SBATCH -o scripts/mega_dense.%j.out
#SBATCH -e scripts/mega_dense.%j.err

# MEGA DENSE: Pack as many experiments as possible per GPU.
# Each experiment uses ~1.3GB VRAM. GH200 has 97GB = ~70 experiments per GPU.
# We run 10 per GPU = 40 total experiments.
#
# Experiments: 5 methods × 5 seeds × 1 task_set = 25 experiments
# + 5 methods × 3 seeds on reach tasks = 15 more
# Total: 40 experiments on 4 GPUs (10 per GPU)

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

echo "MEGA DENSE experiment at $(date)"
echo "Node: $(hostname)"

TASKS=cw_learnable
STEPS=250000
ENVS=256

# GPU 0: finetune (5 seeds CW) + finetune (3 seeds reach) = 8 experiments
for SEED in 42 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/mega_finetune_cw_s${SEED}.json \
        > results/mega_finetune_cw_s${SEED}.log 2>&1 &
done
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
        --method finetune --tasks reach_cycle --steps_per_task 30000 \
        --n_envs 256 --seed $SEED \
        --out results/mega_finetune_reach_s${SEED}.json \
        > results/mega_finetune_reach_s${SEED}.log 2>&1 &
done

# GPU 1: replay (5 seeds CW) + replay (3 seeds reach) = 8 experiments
for SEED in 42 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method replay --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/mega_replay_cw_s${SEED}.json \
        > results/mega_replay_cw_s${SEED}.log 2>&1 &
done
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
        --method replay --tasks reach_cycle --steps_per_task 30000 \
        --n_envs 256 --seed $SEED \
        --out results/mega_replay_reach_s${SEED}.json \
        > results/mega_replay_reach_s${SEED}.log 2>&1 &
done

# GPU 2: ewc (5 seeds CW) + ewc (3 seeds reach) + packnet (2 seeds CW) = 10
for SEED in 42 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method ewc --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/mega_ewc_cw_s${SEED}.json \
        > results/mega_ewc_cw_s${SEED}.log 2>&1 &
done
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method ewc --tasks reach_cycle --steps_per_task 30000 \
        --n_envs 256 --seed $SEED \
        --out results/mega_ewc_reach_s${SEED}.json \
        > results/mega_ewc_reach_s${SEED}.log 2>&1 &
done
for SEED in 3 4; do
    CUDA_VISIBLE_DEVICES=2 python rl/cl_experiment.py \
        --method packnet --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/mega_packnet_cw_s${SEED}.json \
        > results/mega_packnet_cw_s${SEED}.log 2>&1 &
done

# GPU 3: csc (5 seeds CW) + csc (3 seeds reach) + packnet (3 seeds CW) = 11
for SEED in 42 1 2 3 4; do
    CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
        --method csc --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/mega_csc_cw_s${SEED}.json \
        > results/mega_csc_cw_s${SEED}.log 2>&1 &
done
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
        --method csc --tasks reach_cycle --steps_per_task 30000 \
        --n_envs 256 --seed $SEED \
        --out results/mega_csc_reach_s${SEED}.json \
        > results/mega_csc_reach_s${SEED}.log 2>&1 &
done
for SEED in 42 1 2; do
    CUDA_VISIBLE_DEVICES=3 python rl/cl_experiment.py \
        --method packnet --tasks $TASKS --steps_per_task $STEPS \
        --n_envs $ENVS --seed $SEED \
        --out results/mega_packnet_cw_s${SEED}.json \
        > results/mega_packnet_cw_s${SEED}.log 2>&1 &
done

echo "All 40 experiments launched at $(date)"
wait

echo ""
echo "=== CW RESULTS (5 seeds) ==="
for method in finetune replay ewc csc packnet; do
    echo "--- $method ---"
    for s in 42 1 2 3 4; do
        echo -n "  s$s: "
        grep "avg_performance\|avg_forgetting" results/mega_${method}_cw_s${s}.log 2>/dev/null | tr "\n" " " || echo "missing"
        echo
    done
done

echo ""
echo "=== REACH RESULTS (3 seeds) ==="
for method in finetune replay ewc csc; do
    echo "--- $method ---"
    for s in 42 1 2; do
        echo -n "  s$s: "
        grep "avg_performance\|avg_forgetting" results/mega_${method}_reach_s${s}.log 2>/dev/null | tr "\n" " " || echo "missing"
        echo
    done
done
echo "Done at $(date)"
'
