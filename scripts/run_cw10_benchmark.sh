#!/bin/bash
# Full CW10 benchmark: 5 methods × 5 seeds = 25 experiments
# Uses SLURM job chaining (--dependency=afterany) to run in batches of 4.
#
# Each job runs 4 experiments (1 per GPU), ~8.7h each.
# Total: 7 jobs chained, ~3 days wall time.
#
# Usage: bash scripts/run_cw10_benchmark.sh

set -euo pipefail

METHODS=(finetune ewc replay packnet csc)
SEEDS=(42 123 456 789 1337)

# Build list of all 25 (method, seed) pairs
declare -a PAIRS
for m in "${METHODS[@]}"; do
    for s in "${SEEDS[@]}"; do
        PAIRS+=("${m}:${s}")
    done
done

echo "CW10 Benchmark: ${#PAIRS[@]} experiments"
echo "Methods: ${METHODS[*]}"
echo "Seeds: ${SEEDS[*]}"
echo ""

# Submit in batches of 4 (one per GPU)
PREV_JOB=""
BATCH=0
for ((i=0; i<${#PAIRS[@]}; i+=4)); do
    BATCH=$((BATCH+1))

    # Get up to 4 pairs for this batch
    BATCH_PAIRS=()
    for ((j=i; j<i+4 && j<${#PAIRS[@]}; j++)); do
        BATCH_PAIRS+=("${PAIRS[$j]}")
    done

    # Build the sbatch script for this batch
    SCRIPT=$(mktemp /tmp/cw10_batch_XXXX.sbatch)
    cat > "$SCRIPT" << 'HEADER'
#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:4
HEADER
    echo "#SBATCH -J cw10_b${BATCH}" >> "$SCRIPT"
    echo "#SBATCH -o scripts/cw10_b${BATCH}.%j.out" >> "$SCRIPT"
    echo "#SBATCH -e scripts/cw10_b${BATCH}.%j.err" >> "$SCRIPT"
    echo "" >> "$SCRIPT"

    cat >> "$SCRIPT" << 'SETUP'
set -euo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

git pull --ff-only origin main 2>/dev/null || true
SETUP

    echo "" >> "$SCRIPT"
    echo "echo \"Batch ${BATCH}: ${BATCH_PAIRS[*]} at \$(date)\"" >> "$SCRIPT"
    echo "" >> "$SCRIPT"

    # Add each experiment to a different GPU
    GPU=0
    for pair in "${BATCH_PAIRS[@]}"; do
        IFS=':' read -r method seed <<< "$pair"
        echo "CUDA_VISIBLE_DEVICES=${GPU} python rl/cl_experiment.py \\" >> "$SCRIPT"
        echo "    --method ${method} --tasks cw10 \\" >> "$SCRIPT"
        echo "    --steps_per_task 250000 --n_envs 256 --batch_size 128 \\" >> "$SCRIPT"
        echo "    --seed ${seed} --out bench_${method}_s${seed}.json \\" >> "$SCRIPT"
        echo "    > checkpoints/bench_${method}_s${seed}.log 2>&1 &" >> "$SCRIPT"
        echo "" >> "$SCRIPT"
        GPU=$((GPU+1))
    done

    echo "wait" >> "$SCRIPT"
    echo "" >> "$SCRIPT"
    echo "echo \"Batch ${BATCH} done at \$(date)\"" >> "$SCRIPT"

    # Print results summary
    echo "echo \"\"" >> "$SCRIPT"
    echo "echo \"=== BATCH ${BATCH} RESULTS ===\"" >> "$SCRIPT"
    for pair in "${BATCH_PAIRS[@]}"; do
        IFS=':' read -r method seed <<< "$pair"
        echo "echo \"--- ${method} s${seed} ---\"" >> "$SCRIPT"
        echo "grep \"avg_performance\|avg_forgetting\|DONE\" checkpoints/bench_${method}_s${seed}.log 2>/dev/null | tail -4" >> "$SCRIPT"
        echo "echo" >> "$SCRIPT"
    done

    echo "'" >> "$SCRIPT"  # close the uenv bash -lc '...'

    # Submit with dependency on previous job
    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch "$SCRIPT" 2>&1 | grep -o '[0-9]*')
    else
        JOB_ID=$(sbatch --dependency=afterany:${PREV_JOB} "$SCRIPT" 2>&1 | grep -o '[0-9]*')
    fi

    PREV_JOB=$JOB_ID
    echo "Batch ${BATCH}: ${BATCH_PAIRS[*]} → job ${JOB_ID}"

    rm "$SCRIPT"
done

echo ""
echo "All ${BATCH} batches submitted. Chain starts with first available slot."
echo "Monitor: squeue -u \$USER"
echo "Cancel all: scancel ${PREV_JOB} (cancels chain)"
