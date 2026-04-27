#!/bin/bash
#SBATCH -A a0155
#SBATCH -p normal
#SBATCH -t 12:00:00
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=288
#SBATCH --gres=gpu:4
#SBATCH -J sup_csc
#SBATCH -o scripts/sup_csc.%j.out
#SBATCH -e scripts/sup_csc.%j.err

# Supervised CL scaling sweep on Clariden 4xGH200.
# Methods x num_tasks x seeds; ~6 experiments per GPU (ResNet-18 uses ~2GB).
# Default: 5 methods x {10,20,50}-task Split CIFAR-100 x 3 seeds = 45 jobs.
# Override via env: METHODS, NUM_TASKS, SEEDS, EPOCHS.

set -uo pipefail
cd /capstor/scratch/cscs/$USER/CSC

METHODS=${METHODS:-"finetune replay der csc ewc"}
NUM_TASKS=${NUM_TASKS:-"10 20"}
SEEDS=${SEEDS:-"42 123 456"}
EPOCHS=${EPOCHS:-50}
DATASET=${DATASET:-cifar100}
DATA_ROOT=${DATA_ROOT:-/capstor/scratch/cscs/mkalcher/datasets/cifar100}

uenv run prgenv-gnu/24.11:v2 --view=modules -- bash -lc "
set -uo pipefail
module load python/3.12.5
source \"\$HOME/venvs/pt29/bin/activate\"
export PYTHONNOUSERSITE=1

git pull --ff-only origin main 2>/dev/null || true
mkdir -p checkpoints

# Pre-download dataset once so parallel launches don't race on partial writes.
echo \"Pre-downloading dataset $DATASET to $DATA_ROOT (idempotent)...\"
if [ \"$DATASET\" = \"cifar100\" ]; then
    python -c \"import torchvision; torchvision.datasets.CIFAR100(root='$DATA_ROOT', train=True, download=True); torchvision.datasets.CIFAR100(root='$DATA_ROOT', train=False, download=True)\"
elif [ \"$DATASET\" = \"pmnist\" ]; then
    python -c \"import torchvision; torchvision.datasets.MNIST(root='$DATA_ROOT', train=True, download=True); torchvision.datasets.MNIST(root='$DATA_ROOT', train=False, download=True)\"
fi

echo \"GPU supervised sweep at \$(date) [dataset=$DATASET]\"
echo \"Methods: $METHODS\"; echo \"Num tasks: $NUM_TASKS\"; echo \"Seeds: $SEEDS\"

PIDS=()
i=0
for method in $METHODS; do
    case \$method in
        finetune)  args=\"\";;
        replay)    args=\"--replay_per_task 200 --replay_ratio 0.5\";;
        der)       args=\"--replay_per_task 200 --replay_ratio 0.5 --der_alpha 0.5 --der_beta 1.0\";;
        ewc)       args=\"--ewc_lambda 1000\";;
        csc)       args=\"--gamma_comp 0.001 --soft_beta 1.0 --replay_per_task 200 --replay_ratio 0.5 --der_alpha 0.5\";;
        packnet)   args=\"--prune 0.75 --retrain_epochs 10\";;
        *)         args=\"\";;
    esac
    for nt in $NUM_TASKS; do
        for seed in $SEEDS; do
            gpu=\$((i % 4))
            tag=\"_t\${nt}\"
            log=\"checkpoints/sup_${DATASET}_\${method}_t\${nt}_s\${seed}.log\"
            echo \"[GPU \$gpu] \${method} t\${nt} s\${seed} -> \${log}\"
            CUDA_VISIBLE_DEVICES=\$gpu python -u experiments/run_supervised.py \\
                --dataset $DATASET --method \$method --num_tasks \$nt \\
                --epochs_per_task $EPOCHS \\
                --seed \$seed \$args --data_root $DATA_ROOT \\
                > \"\$log\" 2>&1 &
            PIDS+=(\$!)
            i=\$((i+1))
        done
    done
done

n=\${#PIDS[@]}
echo \"\${n} experiments launched on 4 GPUs\"
for pid in \"\${PIDS[@]}\"; do
    wait \$pid || echo \"  pid \$pid exited non-zero\"
done

echo \"Done at \$(date)\"
echo
for f in checkpoints/sup_*.json; do
    [ -f \"\$f\" ] || continue
    method=\$(basename \"\$f\" | sed 's/sup_\\\\([^_]*\\\\)_.*/\\\\1/')
    line=\$(python -c \"import json;d=json.load(open('\$f'));print(f\\\"{d['final_avg']*100:.2f}%  bwt={d['bwt']*100:+.2f}\\\")\")
    echo \"\$f : \$line\"
done
"
