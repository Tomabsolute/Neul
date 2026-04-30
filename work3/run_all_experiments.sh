#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
EPOCHS=${EPOCHS:-20}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_SAMPLES=${NUM_SAMPLES:-6000}
NUM_WORKERS=${NUM_WORKERS:-2}

$PYTHON work3/train_cgan.py \
  --dataset fashion-mnist \
  --download \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-samples "$NUM_SAMPLES" \
  --num-workers "$NUM_WORKERS"

$PYTHON work3/infer.py \
  --checkpoint work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt \
  --output work3/results/figures/inference_grid.png \
  --repeat 8
