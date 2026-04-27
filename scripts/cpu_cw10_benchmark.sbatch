#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:0
#SBATCH -J cpu_cw10
#SBATCH -o scripts/cpu_cw10.%j.out
#SBATCH -e scripts/cpu_cw10.%j.err

# CPU CW10 benchmark — full method coverage with per-task checkpointing.
# Each python process pins to 1 ARM core (OMP_NUM_THREADS=1). Resume-aware:
# rerun the script and each experiment resumes from its checkpoint.
# Submit multiple times to chain past the 12h slot.

set -uo pipefail            # NB: no -e — one failed experiment shouldn't abort the rest
cd /capstor/scratch/cscs/$USER/CSC

# Override defaults via env: STEPS_PER_TASK=500000 sbatch ...
STEPS_PER_TASK=${STEPS_PER_TASK:-1000000}
TASKS=${TASKS:-cw10}

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc "
set -uo pipefail
module load python/3.12.5
source \"\$HOME/venvs/pt29/bin/activate\"
export PYTHONNOUSERSITE=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
# Force CPU even on GH200 nodes: GPUs are visible from the node even when
# --gres=gpu:0, and torch.cuda.is_available() returning True silently routes
# us onto shared GPUs at terrible throughput (and triggers a CUDA-only NaN
# in our quantize() that we have not yet diagnosed).
export CUDA_VISIBLE_DEVICES=\"\"

git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

echo \"CPU CW10 benchmark at \$(date)  STEPS_PER_TASK=$STEPS_PER_TASK TASKS=$TASKS\"

# (method, cl_reg_coef) - CW paper defaults
METHODS=(\"finetune:0\" \"l2:1e5\" \"ewc:1e4\" \"mas:1e4\" \"replay:0\" \"csc:0\")
SEEDS=(42 123 456 789 1234)
PIDS=()

for ms in \"\${METHODS[@]}\"; do
    method=\"\${ms%%:*}\"
    coef=\"\${ms##*:}\"
    for seed in \"\${SEEDS[@]}\"; do
        ckpt=\"checkpoints/cw_\${method}_$TASKS\_s\${seed}.pt\"
        log=\"checkpoints/cpu_\${method}_s\${seed}.log\"
        echo \"Launching: \${method} seed=\${seed} (resume=\${ckpt})\"
        python rl/continual_world.py \\
            --method \"\$method\" \\
            --steps_per_task $STEPS_PER_TASK \\
            --seed \"\$seed\" \\
            --cl_reg_coef \"\$coef\" \\
            --ckpt \"\$ckpt\" \\
            --tasks $TASKS \\
            >> \"\$log\" 2>&1 &
        PIDS+=(\$!)
    done
done

n_jobs=\${#PIDS[@]}
echo \"\${n_jobs} experiments launched on \$(nproc) cores\"
echo \"Waiting...\"
# Wait individually so a single failure doesn't abort the rest
for pid in \"\${PIDS[@]}\"; do
    wait \$pid || echo \"  pid \$pid exited non-zero\"
done

echo
echo '=========================================='
echo 'CPU CW10 BENCHMARK SUMMARY'
echo '=========================================='
for ms in \"\${METHODS[@]}\"; do
    method=\"\${ms%%:*}\"
    for seed in \"\${SEEDS[@]}\"; do
        echo \"--- \${method} s\${seed} ---\"
        tail -12 \"checkpoints/cpu_\${method}_s\${seed}.log\" 2>/dev/null | head -12
        echo
    done
done
echo \"Done at \$(date)\"
"
