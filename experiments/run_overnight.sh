#!/bin/bash
cd /mnt/c/Users/Maxi/Documents/Code/continual_learning

echo "====================================================================="
echo "OVERNIGHT EXPERIMENTS - Started $(date)"
echo "====================================================================="

# =====================================================
# 1. LEARNING RATE CONTROL (most critical)
# =====================================================

echo ""
echo "===== 1a. Replay-only r=200 lr=3.33e-4 (lr/3) ====="
python3 baselines/replay_only.py --num_tasks 10 --epochs_per_task 50 --replay_per_task 200 --lr 3.33e-4 --batch_size 128

echo ""
echo "===== 1b. Replay-only r=200 lr=1.11e-4 (lr/9) ====="
python3 baselines/replay_only.py --num_tasks 10 --epochs_per_task 50 --replay_per_task 200 --lr 1.11e-4 --batch_size 128

echo ""
echo "===== 1c. Replay-only r=200 lr=6.67e-5 (lr/15) ====="
python3 baselines/replay_only.py --num_tasks 10 --epochs_per_task 50 --replay_per_task 200 --lr 6.67e-5 --batch_size 128

echo ""
echo "===== 1d. Replay-only r=500 lr=1.11e-4 (lr/9) ====="
python3 baselines/replay_only.py --num_tasks 10 --epochs_per_task 50 --replay_per_task 500 --lr 1.11e-4 --batch_size 128

# =====================================================
# 2. 50-TASK SCALING EXPERIMENT
# =====================================================

echo ""
echo "===== 2a. Soft CSC 50 tasks ====="
python3 experiments/run_hybrid.py --variant soft --beta 1.0 --gamma 0.001 --replay_per_task 200 --num_tasks 50 --epochs_per_task 20

echo ""
echo "===== 2b. Replay-only 50 tasks ====="
python3 baselines/replay_only.py --num_tasks 50 --epochs_per_task 20 --replay_per_task 200 --batch_size 128

echo ""
echo "===== 2c. PackNet 50 tasks ====="
python3 baselines/packnet.py --num_tasks 50 --epochs_per_task 20 --prune_ratio 0.75 --batch_size 128

# =====================================================
# 3. IMPROVED IMPORTANCE DIFFERENTIATION
# =====================================================

echo ""
echo "===== 3a. Soft CSC g=0.003 b=1.0 r=200 ====="
python3 experiments/run_hybrid.py --variant soft --beta 1.0 --gamma 0.003 --replay_per_task 200

echo ""
echo "===== 3b. Soft CSC g=0.005 b=1.0 r=200 ====="
python3 experiments/run_hybrid.py --variant soft --beta 1.0 --gamma 0.005 --replay_per_task 200

echo ""
echo "===== 3c. Soft CSC g=0.005 b=3.0 r=200 ====="
python3 experiments/run_hybrid.py --variant soft --beta 3.0 --gamma 0.005 --replay_per_task 200

echo ""
echo "===== 3d. Soft CSC relative scaling b=1.0 g=0.01 r=200 ====="
python3 experiments/run_hybrid.py --variant soft --beta 1.0 --gamma 0.01 --replay_per_task 200 --relative_scaling

# =====================================================
# 4. BETA SWEEP AT r=500
# =====================================================

echo ""
echo "===== 4a. Soft CSC b=0.5 g=0.001 r=500 ====="
python3 experiments/run_hybrid.py --variant soft --beta 0.5 --gamma 0.001 --replay_per_task 500

echo ""
echo "===== 4b. Soft CSC b=3.0 g=0.001 r=500 ====="
python3 experiments/run_hybrid.py --variant soft --beta 3.0 --gamma 0.001 --replay_per_task 500

echo ""
echo "====================================================================="
echo "ALL OVERNIGHT EXPERIMENTS DONE - $(date)"
echo "====================================================================="
