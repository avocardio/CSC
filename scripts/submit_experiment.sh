#!/bin/bash
# Submit a single CW10 experiment as a chain of 12h jobs with checkpoint-resume.
# Each job trains ~3 tasks (1M steps/task at ~80 sps = 3.5h/task).
#
# Usage: bash scripts/submit_experiment.sh <method> <seed> [extra_args] [run_name]
# Example: bash scripts/submit_experiment.sh csc 42 "--gamma_comp 0.01" "csc_gc001"

set -euo pipefail

METHOD=${1:?Usage: submit_experiment.sh <method> <seed> [extra_args] [run_name]}
SEED=${2:?Usage: submit_experiment.sh <method> <seed> [extra_args] [run_name]}
EXTRA=${3:-}
RUN_NAME=${4:-${METHOD}_s${SEED}}

N_JOBS=4
echo "Submitting: $RUN_NAME ($N_JOBS chained jobs)"

PREV_JOB=""
for JOB_NUM in $(seq 1 $N_JOBS); do
    SCRIPT=$(mktemp /tmp/cw10_${RUN_NAME}_j${JOB_NUM}_XXXX.sbatch)
    cat > "$SCRIPT" << EOF
#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --gres=gpu:1
#SBATCH -J ${RUN_NAME}_j${JOB_NUM}
#SBATCH -o scripts/${RUN_NAME}_j${JOB_NUM}.%j.out
#SBATCH -e scripts/${RUN_NAME}_j${JOB_NUM}.%j.err

set -euo pipefail
cd /capstor/scratch/cscs/\$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc '
set -euo pipefail
module load python/3.12.5 cuda/12.6.0
source "\$HOME/venvs/pt29/bin/activate"
export PYTHONNOUSERSITE=1 CUDA_DEVICE_ORDER=PCI_BUS_ID PYTHONUNBUFFERED=1
mkdir -p checkpoints

echo "${RUN_NAME} job ${JOB_NUM}/${N_JOBS} at \$(date)"

RESUME_ARG=""
for t in 9 8 7 6 5 4 3 2 1 0; do
    CKPT="checkpoints/${RUN_NAME}_task\${t}.pt"
    if [ -f "\$CKPT" ]; then
        RESUME_ARG="--resume \$CKPT"
        echo "Resuming from \$CKPT"
        break
    fi
done

python rl/cl_experiment.py \\
    --method ${METHOD} --tasks cw10 \\
    --steps_per_task 1000000 --n_envs 256 --batch_size 128 \\
    --seed ${SEED} --out ${RUN_NAME}.json \\
    --run_name ${RUN_NAME} \\
    ${EXTRA} \\
    \$RESUME_ARG

echo "Job ${JOB_NUM} done at \$(date)"
'
EOF

    if [ -z "$PREV_JOB" ]; then
        JOB_ID=$(sbatch "$SCRIPT" 2>&1 | grep -o '[0-9]*')
    else
        JOB_ID=$(sbatch --dependency=afterany:${PREV_JOB} "$SCRIPT" 2>&1 | grep -o '[0-9]*')
    fi
    PREV_JOB=$JOB_ID
    echo "  Job ${JOB_NUM}/${N_JOBS}: $JOB_ID"
    rm "$SCRIPT"
done
