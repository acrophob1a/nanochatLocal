#!/bin/bash
# 从已有 SFT checkpoint 续训，强化中文（chinese_conversations.jsonl）
# 用法：bash runs/chinese_sft.sh
# 多卡：NPROC=4 bash runs/chinese_sft.sh

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$ROOT_DIR/nanochat_data}"
export WANDB_MODE="${WANDB_MODE:-disabled}"
# --run dummy 跳过 wandb；checkpoint meta 里仍可通过环境变量 CHINESE_SFT_LABEL 区分本次实验
export CHINESE_SFT_LABEL="${CHINESE_SFT_LABEL:-chinese_sft}"

if [ ! -f "$NANOCHAT_BASE_DIR/chinese_conversations.jsonl" ]; then
  echo "错误: 未找到 $NANOCHAT_BASE_DIR/chinese_conversations.jsonl"
  exit 1
fi

if [ -d ".venv" ]; then
  source .venv/bin/activate
fi

NPROC="${NPROC:-1}"
MODEL_TAG="${MODEL_TAG:-d26}"
MODEL_STEP="${MODEL_STEP:-747}"
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-65536}"
NUM_ITERATIONS="${NUM_ITERATIONS:-400}"
INIT_LR_FRAC="${INIT_LR_FRAC:-0.2}"

COMMON_ARGS=(
  --resume-from sft
  --model-tag "$MODEL_TAG"
  --model-step "$MODEL_STEP"
  --device-batch-size "$DEVICE_BATCH_SIZE"
  --total-batch-size "$TOTAL_BATCH_SIZE"
  --num-iterations "$NUM_ITERATIONS"
  --init-lr-frac "$INIT_LR_FRAC"
  --eval-every 50
  --eval-tokens 524288
  --run dummy
)

echo "NANOCHAT_BASE_DIR=$NANOCHAT_BASE_DIR"
echo "从 sft/$MODEL_TAG step $MODEL_STEP 续训，中文 SFT，${NUM_ITERATIONS} 步"

if [ "$NPROC" -gt 1 ]; then
  torchrun --standalone --nproc_per_node="$NPROC" -m scripts.chat_sft -- "${COMMON_ARGS[@]}"
else
  python -m scripts.chat_sft "${COMMON_ARGS[@]}"
fi

echo ""
echo "训练完成。查看新 checkpoint:"
echo "  ls $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/model_*.pt"
echo "推理示例（将 STEP 换成最新步数）:"
echo "  python -m scripts.chat_cli -i sft -g $MODEL_TAG -s STEP -p \"你好，你是谁？\""
