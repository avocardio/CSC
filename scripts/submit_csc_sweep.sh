#!/bin/bash
# CSC hyperparameter sweep on CW10.
# Sweep gamma_comp (compression weight) and grad_scale_beta (gradient scaling).
# Uses seed=42 for all sweep configs, then best config runs with 5 seeds.
#
# Usage: bash scripts/submit_csc_sweep.sh

set -euo pipefail

SEED=42

# Sweep grid:
# gamma_comp: controls how much compression matters (higher = more pruning pressure)
# grad_scale_beta: controls how much important weights are protected (higher = stronger protection)
GAMMA_COMPS=(0.001 0.01 0.05 0.1)
BETAS=(0.5 1.0 2.0 5.0)

echo "====================================="
echo "CSC Hyperparameter Sweep"
echo "gamma_comp: ${GAMMA_COMPS[*]}"
echo "grad_scale_beta: ${BETAS[*]}"
echo "Total configs: $((${#GAMMA_COMPS[@]} * ${#BETAS[@]}))"
echo "====================================="
echo ""

for gc in "${GAMMA_COMPS[@]}"; do
    for beta in "${BETAS[@]}"; do
        TAG="csc_gc${gc}_b${beta}"
        echo "--- ${TAG} ---"
        bash scripts/submit_experiment.sh "csc" "$SEED" \
            "--gamma_comp $gc --grad_scale_beta $beta"
        echo ""
    done
done

echo "Sweep submitted. ${#GAMMA_COMPS[@]} × ${#BETAS[@]} = $((${#GAMMA_COMPS[@]} * ${#BETAS[@]})) configurations."
echo "Collect results with: python scripts/collect_results.py"
