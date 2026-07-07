#!/bin/bash
# 仅中文语料 SFT（扩充版，5061 条 Belle+nanochat 数据）
# 用法：bash runs/chinese_only_sft_v2.sh
# 从 step 747 续训：MODEL_STEP=747 bash runs/chinese_only_sft_v2.sh
#
# 与 v1 的区别：
#   - 数据量 75 → 5061 条，chinese-repeats 20 → 1（无需放大）
#   - epoch 20 → 3（数据量充足，3 epoch 足够收敛）
#   - 适配单 GPU：降低 total-batch-size，加快评估频率

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
MODEL_STEP="${MODEL_STEP:-747}"
CHINESE_REPEATS="${CHINESE_REPEATS:-1}"        # 5061 条已足够，无需重复
NUM_EPOCHS="${NUM_EPOCHS:-3}"                  # 数据量充足，3 epoch 足够
DEVICE_BATCH_SIZE="${DEVICE_BATCH_SIZE:-2}"
TOTAL_BATCH_SIZE="${TOTAL_BATCH_SIZE:-32768}"  # 单 GPU 降低累积
INIT_LR_FRAC="${INIT_LR_FRAC:-0.1}"            # 数据量大，学习率更保守
EVAL_EVERY="${EVAL_EVERY:-50}"
EVAL_TOKENS="${EVAL_TOKENS:-131072}"

echo "============================================================"
echo "中文 SFT v2（扩充数据版）"
echo "  起点:           sft/$MODEL_TAG step $MODEL_STEP"
echo "  数据:           chinese_conversations.jsonl (5061 条)"
echo "  chinese-repeats: $CHINESE_REPEATS"
echo "  num-epochs:     $NUM_EPOCHS"
echo "  device-batch:   $DEVICE_BATCH_SIZE"
echo "  total-batch:    $TOTAL_BATCH_SIZE tokens"
echo "  init-lr-frac:   $INIT_LR_FRAC"
echo "  eval-every:     $EVAL_EVERY 步"
echo "============================================================"

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
  --eval-every "$EVAL_EVERY" \
  --eval-tokens "$EVAL_TOKENS" \
  --run dummy

echo ""
echo "训练完成。新 checkpoint:"
echo "  ls -lt $NANOCHAT_BASE_DIR/chatsft_checkpoints/$MODEL_TAG/model_*.pt | head"
echo ""
echo "推理测试:"
echo "  python -m scripts.chat_cli -i sft -g $MODEL_TAG -s STEP -p \"你好，请介绍一下自己\""
