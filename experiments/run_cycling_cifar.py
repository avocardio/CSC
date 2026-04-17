"""Cycling CIFAR-100: repeat the incremental class sequence multiple times.

The original Nature plasticity paper showed plasticity loss on VERY long sequences.
Our 4000-epoch single pass may have been too short. Here we cycle through the
class sequence multiple times to induce genuine plasticity loss.

Setup: 100 classes, add 5 at a time (same as before), but after reaching 100,
reset to 5 and start over. Do this for N cycles.
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
import numpy as np
import argparse
import time
from torch.utils.data import DataLoader, Subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--method', type=str, default='standard',
                        choices=['standard', 'compression', 'random', 'l2_reset'])
    parser.add_argument('--n_cycles', type=int, default=3)
    parser.add_argument('--epochs_per_phase', type=int, default=50)
    parser.add_argument('--lr', type=float, default=0.01)
    parser.add_argument('--replacement_rate', type=float, default=0.001)
    parser.add_argument('--gamma', type=float, default=0.01)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    print(f"Cycling CIFAR-100: method={args.method}, cycles={args.n_cycles}")
    print(f"Total phases: {args.n_cycles * 20}, total epochs: {args.n_cycles * 20 * args.epochs_per_phase}")

    # This will be a long experiment. Just print the setup for now.
    # Full implementation follows the same pattern as run_plasticity.py
    # but cycles through class additions multiple times.

    print("Setup complete. Run with GPU when available.")


if __name__ == '__main__':
    main()
