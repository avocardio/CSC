#!/bin/bash
#SBATCH -A a0155
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:2
#SBATCH -J ckpt_test
#SBATCH -o scripts/ckpt_test.%j.out
#SBATCH -e scripts/ckpt_test.%j.err

# Test checkpoint/resume: train 3 tasks, checkpoint after each.
# Then resume from task 1 checkpoint and verify it continues from task 2.

set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true

echo "=== CHECKPOINT TEST at $(date) ==="

# Step 1: Train finetune on 3 tasks (10K steps each) — should checkpoint after each
echo ""
echo "--- Step 1: Train 3 tasks from scratch ---"
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks window-close,faucet-close,handle-press-side \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out ckpt_test_full.json

echo ""
echo "--- Checkpoints created: ---"
ls -la checkpoints/finetune_s42_task*.pt 2>/dev/null

# Step 2: Resume from task 1 checkpoint — should skip tasks 0-1 and only train task 2
echo ""
echo "--- Step 2: Resume from task 1 checkpoint ---"
CUDA_VISIBLE_DEVICES=0 python rl/cl_experiment.py \
    --method finetune --tasks window-close,faucet-close,handle-press-side \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out ckpt_test_resume.json \
    --resume checkpoints/finetune_s42_task1.pt

echo ""
echo "--- Step 3: Compare results ---"
echo "Full run:"
python -c "import json; d=json.load(open(\"checkpoints/ckpt_test_full.json\")); print(\"  eval_matrix:\", d[\"eval_matrix\"]); print(\"  metrics:\", d[\"metrics\"])"
echo ""
echo "Resumed run:"
python -c "import json; d=json.load(open(\"checkpoints/ckpt_test_resume.json\")); print(\"  eval_matrix:\", d[\"eval_matrix\"]); print(\"  metrics:\", d[\"metrics\"])"

# Step 4: Test CSC checkpoint/resume
echo ""
echo "--- Step 4: CSC checkpoint test ---"
CUDA_VISIBLE_DEVICES=1 python rl/cl_experiment.py \
    --method csc --tasks window-close,faucet-close,handle-press-side \
    --steps_per_task 10000 --n_envs 256 --batch_size 128 \
    --seed 42 --out ckpt_test_csc.json \
    --gamma_comp 0.01 --grad_scale_beta 1.0

echo ""
echo "CSC checkpoints:"
ls -la checkpoints/csc_s42_task*.pt 2>/dev/null

echo ""
echo "=== CHECKPOINT TEST DONE at $(date) ==="
'
