#!/bin/bash
# Run all CL methods sequentially
set -e

cd /mnt/c/Users/Maxi/Documents/Code/continual_learning
export LD_LIBRARY_PATH=/usr/lib/wsl/lib:$LD_LIBRARY_PATH
export PYTHONUNBUFFERED=1

PY=/home/maxi/.venvs/cuda/bin/python
STEPS=${1:-30000}
SEED=${2:-42}
TASKS=${3:-reach_cycle}

for method in finetune ewc replay compression csc; do
    echo ""
    echo "============================================"
    echo " METHOD: $method"
    echo "============================================"
    $PY rl/cl_experiment.py \
        --method $method \
        --tasks $TASKS \
        --steps_per_task $STEPS \
        --n_envs 64 \
        --seed $SEED \
        --out "cl_${method}_${TASKS}_s${SEED}.json" 2>&1 || echo "  [$method failed]"
done

echo ""
echo "All methods done. Results in checkpoints/"
ls -1 checkpoints/cl_*.json 2>/dev/null | tail -10
