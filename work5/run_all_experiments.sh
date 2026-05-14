#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
SCRATCH_STEPS=${SCRATCH_STEPS:-300}
SFT_STEPS=${SFT_STEPS:-100}
DPO_STEPS=${DPO_STEPS:-80}
MODEL_NAME=${MODEL_NAME:-unsloth/Qwen3-0.6B-unsloth-bnb-4bit}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

"$PYTHON" work5/scripts/prepare_sunzi_dataset.py \
  --output-dir work5/data

"$PYTHON" work5/scripts/train_from_scratch.py \
  --train-file work5/data/pretrain.txt \
  --output-dir work5/results/scratch \
  --max-steps "$SCRATCH_STEPS"

"$PYTHON" work5/scripts/train_qwen3_sft.py \
  --model-name "$MODEL_NAME" \
  --data work5/data/sft.jsonl \
  --output-dir work5/results/qwen3_sft \
  --max-steps "$SFT_STEPS"

"$PYTHON" work5/scripts/train_qwen3_dpo.py \
  --model-name "$MODEL_NAME" \
  --data work5/data/dpo.jsonl \
  --sft-adapter work5/results/qwen3_sft \
  --output-dir work5/results/qwen3_dpo \
  --max-steps "$DPO_STEPS"

"$PYTHON" work5/scripts/infer_qwen3.py \
  --model-name "$MODEL_NAME" \
  --adapter work5/results/qwen3_dpo \
  --prompt "今有物不知其數，三三數之剩二，五五數之剩三，七七數之剩二。問物幾何？" \
  > work5/results/qwen3_dpo/sample_generation.txt

echo "Work5 results written to work5/results"
