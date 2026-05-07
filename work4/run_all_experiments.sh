#!/usr/bin/env bash
set -euo pipefail

PYTHON=${PYTHON:-python3}
LM_EPOCHS=${LM_EPOCHS:-300}
EMB_EPOCHS=${EMB_EPOCHS:-40}
BATCH_SIZE=${BATCH_SIZE:-32}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

"$PYTHON" work4/train_lm.py \
  --epochs "$LM_EPOCHS" \
  --batch-size "$BATCH_SIZE" \
  --seq-len 384 \
  --embedding-dim 256 \
  --hidden-dim 512 \
  --dropout 0.1 \
  --sample-temperature 0

"$PYTHON" work4/train_word2vec.py \
  --epochs "$EMB_EPOCHS"

"$PYTHON" work4/query_embeddings.py \
  --checkpoint work4/results/word2vec/skipgram_embeddings.pt \
  --vocab work4/results/word2vec/vocab.json \
  --queries "田,畝,分,率,粟,米,步,尺" \
  > work4/results/word2vec/manual_neighbors.md

"$PYTHON" work4/generate_answer.py \
  --prompt "〔示例〕今有田廣十五步，從十六步。問為田幾何？" \
  --checkpoint work4/results/lstm_lm/best_lstm_lm.pt \
  --vocab work4/results/lstm_lm/vocab.json \
  > work4/results/lstm_lm/manual_generation.txt

echo "Results written to work4/results"
