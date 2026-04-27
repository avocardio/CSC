#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 4:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J sup_csc_redo
#SBATCH -o scripts/sup_csc_redo.%j.out
#SBATCH -e scripts/sup_csc_redo.%j.err

# Resubmit ONLY the seeds that crashed in job 1955641 due to the
# concurrent-download race (csc/der/replay seed 42, der seed 456).
# The cached dataset from the earlier successful runs lets these complete cleanly.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc "
set -uo pipefail
module load python/3.12.5
source \"\$HOME/venvs/pt29/bin/activate\"
export PYTHONNOUSERSITE=1

git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

echo \"Resubmit failed seeds at \$(date)\"

# (method, seed, gpu) tuples — manually scheduled to use 4 GPUs
declare -a JOBS=(
  \"csc 42 0\"
  \"der 42 1\"
  \"der 456 2\"
  \"replay 42 3\"
)

PIDS=()
for j in \"\${JOBS[@]}\"; do
    read method seed gpu <<< \"\$j\"
    case \$method in
        replay) args=\"--replay_per_task 200 --replay_ratio 0.5\";;
        der)    args=\"--replay_per_task 200 --replay_ratio 0.5 --der_alpha 0.5 --der_beta 1.0\";;
        csc)    args=\"--gamma_comp 0.001 --soft_beta 1.0 --replay_per_task 200 --replay_ratio 0.5 --der_alpha 0.5\";;
    esac
    log=\"checkpoints/sup_\${method}_t10_s\${seed}.log\"
    echo \"[GPU \$gpu] \${method} t10 s\${seed} -> \${log}\"
    CUDA_VISIBLE_DEVICES=\$gpu python -u experiments/run_supervised.py \\
        --method \$method --num_tasks 10 --epochs_per_task 50 \\
        --seed \$seed \$args \\
        --data_root /capstor/scratch/cscs/mkalcher/datasets/cifar100 \\
        > \"\$log\" 2>&1 &
    PIDS+=(\$!)
done

for pid in \"\${PIDS[@]}\"; do
    wait \$pid || echo \"  pid \$pid exited non-zero\"
done

echo \"Done at \$(date)\"
"
