#!/usr/bin/env bash
cd /home/user/QT-Net
export XLA_FLAGS="--xla_cpu_multi_thread_eigen=true"
export PYTHONUNBUFFERED=1
# n_train=350 fits in a SINGLE padded batch (batch_size 512): one XLA compile,
# fast epochs, bounded memory -- the multi-batch path recompiles per shape and
# leaks memory on CPU. Runs are checkpointed + resumable per (seed, variant).
SHARED="--max-neighbors 12 --n-train 350 --n-val 120 --batch-size 512"
echo "###### SGNN ######"
.venv/bin/python -u scripts/atomic/experiment_learnable_activations.py \
    --model SGNN $SHARED --epochs 120 --seeds 0 1 2
echo "###### EGNN ######"
# EGNN (e3nn tensor products) is much slower on CPU -> lighter budget.
.venv/bin/python -u scripts/atomic/experiment_learnable_activations.py \
    --model EGNN $SHARED --epochs 100 --seeds 0 1
echo "###### ALL DONE ######"
