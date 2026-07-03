#!/usr/bin/env bash
set -e
cd /home/user/QT-Net
COMMON="--max-neighbors 12 --n-train 1500 --n-val 400 --epochs 100 --seeds 0 1 2 --batch-size 512"
echo "###### SGNN ######"
.venv/bin/python -u scripts/atomic/experiment_learnable_activations.py --model SGNN $COMMON
echo "###### EGNN ######"
.venv/bin/python -u scripts/atomic/experiment_learnable_activations.py --model EGNN $COMMON
echo "###### ALL DONE ######"
