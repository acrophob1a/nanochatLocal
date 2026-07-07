#!/bin/bash
# 仅中文语料 SFT，可反复多 epoch（默认 20 遍池子 × 20 epoch）
# 用法：bash runs/chinese_only_sft.sh
# 从 step 25 续训：MODEL_STEP=25 bash runs/chinese_only_sft.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$ROOT_DIR/nanochat_data}"
export WANDB_MODE="${WANDB_MODE:-disabled}"

if [ ! -f "$NANOCHAT_BASE_DIR/chinese_conversations.jsonl" ]; then
  echo "错误: 未找到 chinese_conversations.jsonl"
  exit 1
fi

[ -d ".venv" ] && source .venv/bin/activate

MODEL_TAG="${MODEL_TAG:-d26}"
MODEL_STEP="${MODEL_STEP:-25}"
CHINESE_REPEATS="${CHINESE_REPEATS:-20}"
NUM_EPOCHS="${NUM_EPOCHS:-20}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
INIT_LR_FRAC="${INIT_LR_FRAC:-0.15}"

python -m scripts.chat_sft \
  --resume-from sft \
  --model-tag "$MODEL_TAG" \
  --model-step "$MODEL_STEP" \
  --chinese-only \
  --chinese-repeats "$CHINESE_REPEATS" \
  --num-epochs "$NUM_EPOCHS" \
  --num-iterations -1 \
  --device-batch-size "$DEVICE_BATCH_SIZE" \
  --total-batch-size "$TOTAL_BATCH_SIZE" \
  --init-lr-frac "$INIT_LR_FRAC" \
  --eval-every 100 \
  --eval-tokens 262144 \
  --run dummy

echo ""
echo "完成。新 checkpoint 步数约为: $((MODEL_STEP)) + (optimizer steps)"
echo "  ls -lt $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/model_*.pt | head"
