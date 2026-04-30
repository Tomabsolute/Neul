#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
EPOCHS=${EPOCHS:-50}
BATCH_SIZE=${BATCH_SIZE:-128}
NUM_SAMPLES=${NUM_SAMPLES:-20000}
NUM_WORKERS=${NUM_WORKERS:-2}
LR=${LR:-0.0002}
Z_DIM=${Z_DIM:-100}
SAMPLE_EVERY=${SAMPLE_EVERY:-5}
REPEAT=${REPEAT:-8}

$PYTHON work3/train_cgan.py \
  --dataset fashion-mnist \
  --download \
  --epochs "$EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --num-samples "$NUM_SAMPLES" \
  --num-workers "$NUM_WORKERS" \
  --lr "$LR" \
  --z-dim "$Z_DIM" \
  --sample-every "$SAMPLE_EVERY"

$PYTHON work3/infer.py \
  --checkpoint work3/results/cgan_fashion_mnist/checkpoints/best_generator.pt \
  --output work3/results/figures/inference_grid.png \
  --repeat "$REPEAT"
