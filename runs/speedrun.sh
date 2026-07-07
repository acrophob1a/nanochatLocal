#!/bin/bash
set -euo pipefail

# This script is configured to train your own GPT-2 grade LLM (pretraining + finetuning)
# It is designed to run on a blank 8XH100 GPU node and takes approximately 3 hours to complete.

# 1) Example launch (simplest):
# bash runs/speedrun.sh
# 2) Example launch in a screen session (because the run takes ~3 hours):
# screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh
# 3) Example launch with wandb logging, but see below for setting up wandb first:
# WANDB_RUN=speedrun screen -L -Logfile runs/speedrun.log -S speedrun bash runs/speedrun.sh

# -----------------------------------------------------------------------
# Disable Conda first so pip/python/curl use system defaults.
# Otherwise curl fails with certificate file from anaconda path (missing in container),
# and uv/venv can't be set up correctly.
# -----------------------------------------------------------------------
unset CONDA_SHLVL CONDA_EXE _CE_CONDA CONDA_PREFIX \
      CONDA_PROMPT_MODIFIER CONDA_PYTHON_EXE CONDA_DEFAULT_ENV 2>/dev/null || true
export PATH="$(echo "$PATH" | tr ':' '\n' | grep -v 'anaconda3' | paste -sd ':' -)" 2>/dev/null || true
# Avoid curl (77) certificate errors when anaconda cert path doesn't exist in job/container
unset SSL_CERT_FILE SSL_CERT_DIR CURL_CA_BUNDLE REQUESTS_CA_BUNDLE 2>/dev/null || true
echo "[INFO] Conda disabled; SSL cert env cleared for curl."

# Ensure we run from project root (job may start in any directory)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$ROOT_DIR"

# Data and intermediate artifacts directory: ./nanochat_data (use local dir instead of ~/.cache/nanochat)
export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="$ROOT_DIR/nanochat_data"
mkdir -p "$NANOCHAT_BASE_DIR"

# -----------------------------------------------------------------------------
# Python venv and dependencies (pyproject.toml); job env may already have torch, we use venv for exact versions

# install uv (if not already installed) and ensure it's on PATH
if ! command -v uv &> /dev/null; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
    export PATH="${HOME}/.local/bin:${PATH}"
fi
# create a .venv in project root (if it doesn't exist)
[ -d ".venv" ] || uv venv
# install all dependencies from pyproject.toml (including torch/torchao via --extra gpu for CUDA)
uv sync --extra gpu
# activate venv so that `python`/`torchrun` use the project's env
source .venv/bin/activate

# -----------------------------------------------------------------------------
# Disable wandb (use dummy run name so scripts skip logging to wandb)
export WANDB_RUN=dummy
export WANDB_MODE=disabled

# -----------------------------------------------------------------------------
# During the course of the run, we will be writing markdown reports to the report/
# directory in the base dir. This command clears it out and writes a header section
# with a bunch of system info and a timestamp that marks the start of the run.
python -m nanochat.report reset

# -----------------------------------------------------------------------------
# Tokenizer

# Download the first ~2B characters of pretraining dataset
# each data shard is ~250M chars
# so we download 2e9 / 250e6 = 8 data shards at this point
# each shard is ~100MB of text (compressed), so this is about ~800MB of data on disk
# look at dev/repackage_data_reference.py for details on how this data was prepared
python -m nanochat.dataset -n 8
# Immediately also kick off downloading more shards in the background while tokenizer trains
# Approximately 350 shards are needed for 10B tokens of data for pretraining.
# The maximum total number of shards available in the entire dataset is 1822.
python -m nanochat.dataset -n 370 &
DATASET_DOWNLOAD_PID=$!
# train the tokenizer with vocab size 2**15 = 32768 on ~2B characters of data
python -m scripts.tok_train
# evaluate the tokenizer (report compression ratio etc.)
python -m scripts.tok_eval

# -----------------------------------------------------------------------------
# Base model (pretraining)
# Pretrain ckpt is saved under: $NANOCHAT_BASE_DIR/base_checkpoints/<tag>/
# with tag = --model-tag if set, else "d<depth>" (e.g. depth=26 => base_checkpoints/d26/).
echo "Waiting for dataset download to complete..."
wait $DATASET_DOWNLOAD_PID

BASE_DEPTH=26
# d26 model. No --fp8: fp8e4nv needs H100; current node supports fp8e4b15/fp8e5 only, use bf16/fp16.
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.base_train -- --depth=$BASE_DEPTH --target-param-data-ratio=8.5 --device-batch-size=16 --run=$WANDB_RUN
# evaluate the model: CORE metric, BPB on train/val, and draw samples
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.base_eval -- --device-batch-size=16

# -----------------------------------------------------------------------------
# SFT (teach the model conversation special tokens, tool use, multiple choice)
# Load from the pretrain ckpt above: $NANOCHAT_BASE_DIR/base_checkpoints/d<depth>/.
# Step is auto-detected (latest model_*.pt) if --model-step is omitted. Batch size 2 to avoid OOM.
#
# If you already have pretrain and want to run ONLY SFT (skip tokenizer + base_train + base_eval),
# run from here: set NANOCHAT_BASE_DIR and activate venv, then run the curl + two python lines below
# (or copy them into a one-off script). Do NOT run the full speedrun.sh from the top.
SFT_MODEL_TAG=d${BASE_DEPTH}
SFT_DEVICE_BATCH_SIZE=2

# download 2.3MB of synthetic identity conversations to impart a personality to nanochat
# see dev/gen_synthetic_data.py for details on how this data was prepared and to get a sense of how you can easily tune it
curl -L -o $NANOCHAT_BASE_DIR/identity_conversations.jsonl https://karpathy-public.s3.us-west-2.amazonaws.com/identity_conversations.jsonl

# run SFT from pretrain ckpt (--model-tag; step auto-detected from base_checkpoints/$SFT_MODEL_TAG/)
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.chat_sft -- \
  --model-tag "$SFT_MODEL_TAG" --device-batch-size="$SFT_DEVICE_BATCH_SIZE" --run=$WANDB_RUN
python -m torch.distributed.run --standalone --nproc_per_node=8 -m scripts.chat_eval -- -i sft

# chat with the model over CLI! Leave out the -p to chat interactively
# python -m scripts.chat_cli -p "Why is the sky blue?"

# even better, chat with your model over a pretty WebUI ChatGPT style
# python -m scripts.chat_web

# -----------------------------------------------------------------------------
# Generate the full report by putting together all the sections
# report.md is the output and will be copied to current directory for convenience
python -m nanochat.report generate
