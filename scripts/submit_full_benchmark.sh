#!/bin/bash
# Submit full CW10 benchmark: 5 methods × 5 seeds = 25 experiments.
# Each experiment is a chain of 4 × 12h jobs with checkpoint-resume.
# Total: 100 jobs, ~48h wall time per experiment.
#
# Usage: bash scripts/submit_full_benchmark.sh

set -euo pipefail

METHODS=(finetune l2 ewc mas replay packnet csc)
SEEDS=(42 123 456 789 1337)

echo "====================================="
echo "CW10 Full Benchmark"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo "Steps/task: 1M"
echo "====================================="
echo ""

for method in "${METHODS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        EXTRA=""
        if [ "$method" = "csc" ]; then
            EXTRA="--gamma_comp 0.01 --grad_scale_beta 1.0"
        fi
        echo "--- ${method} seed=${seed} ---"
        bash scripts/submit_experiment.sh "$method" "$seed" "$EXTRA"
        echo ""
    done
done

echo "All experiments submitted."
echo "Total: ${#METHODS[@]} methods × ${#SEEDS[@]} seeds = $((${#METHODS[@]} * ${#SEEDS[@]})) experiments"
