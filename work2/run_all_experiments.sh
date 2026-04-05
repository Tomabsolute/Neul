#!/usr/bin/env bash
set -euo pipefail

# Minimal work2 runner (Food101 only)
# Optional env vars:
#   PYTHON=python3 BATCH_SIZE=64 NUM_WORKERS=8 AMP=1 MULTI_GPU=1

PYTHON=${PYTHON:-python3}
BATCH_SIZE=${BATCH_SIZE:-64}
NUM_WORKERS=${NUM_WORKERS:-8}
AMP=${AMP:-1}
MULTI_GPU=${MULTI_GPU:-1}

COMMON_ARGS=(
  --data-root ./work2/data
  --out-dir ./work2/results
  --batch-size "$BATCH_SIZE"
  --num-workers "$NUM_WORKERS"
)

if [[ "$AMP" == "1" ]]; then
  COMMON_ARGS+=(--amp)
fi
if [[ "$MULTI_GPU" == "1" ]]; then
  COMMON_ARGS+=(--multi-gpu)
fi

# 1) ResNeXt scratch
"$PYTHON" ./work2/train_flowers102.py \
  --model resnext50_32x4d --mode scratch \
  --scratch-epochs 60 --scratch-lr 0.01 \
  "${COMMON_ARGS[@]}"

# 2) ResNeXt finetune
"$PYTHON" ./work2/train_flowers102.py \
  --model resnext50_32x4d --mode finetune \
  --freeze-epochs 10 --finetune-epochs 40 \
  --head-lr 0.001 --backbone-lr 0.0001 \
  "${COMMON_ARGS[@]}"

# 3) DenseNet scratch
"$PYTHON" ./work2/train_flowers102.py \
  --model densenet121 --mode scratch \
  --scratch-epochs 60 --scratch-lr 0.01 \
  "${COMMON_ARGS[@]}"

# 4) DenseNet finetune
"$PYTHON" ./work2/train_flowers102.py \
  --model densenet121 --mode finetune \
  --freeze-epochs 10 --finetune-epochs 40 \
  --head-lr 0.001 --backbone-lr 0.0001 \
  "${COMMON_ARGS[@]}"

# Aggregate compare figures
"$PYTHON" ./work2/visualize_results.py \
  --summary ./work2/results/summary.csv \
  --out-dir ./work2/results/figures

echo "All experiments finished."
echo "Summary: ./work2/results/summary.csv"
echo "Figures: ./work2/results/figures"
