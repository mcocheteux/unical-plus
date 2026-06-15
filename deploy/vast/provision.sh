#!/usr/bin/env bash
#
# Provisioning / onstart script for a vast.ai instance to prepare UniCal training.
#
# What it does (idempotent, safe to re-run):
#   1. installs system deps (git/curl/wget/unzip)
#   2. installs uv
#   3. clones/updates this repo
#   4. `uv sync` (installs torch + CUDA wheels, lightning, transformers, ...)
#   5. downloads the KITTI raw drives needed by the chosen experiment
#
# It deliberately does NOT start training, so you can inspect the box first.
#
# Use it either as the vast.ai onstart script (`--onstart provision.sh`) or run it
# manually after you SSH in. Configure via environment variables:
#   REPO_URL    git remote to clone           (default: this repo's origin)
#   BRANCH      branch to check out            (default: main)
#   CODE_DIR    where to put the code          (default: /workspace/unical)
#   DATA_DIR    where to put the KITTI dataset (default: /workspace/kitti_raw)
#   KITTI_MODE  'default' (full) or 'debug'    (default: default)
#
set -euo pipefail

REPO_URL="${REPO_URL:-https://github.com/mcocheteux/unical.git}"
BRANCH="${BRANCH:-main}"
CODE_DIR="${CODE_DIR:-/workspace/unical}"
DATA_DIR="${DATA_DIR:-/workspace/kitti_raw}"
KITTI_MODE="${KITTI_MODE:-default}"

echo "==> [1/5] system dependencies"
if command -v apt-get >/dev/null 2>&1; then
  export DEBIAN_FRONTEND=noninteractive
  apt-get update -y
  apt-get install -y --no-install-recommends git curl wget unzip ca-certificates
fi

echo "==> [2/5] uv"
if ! command -v uv >/dev/null 2>&1; then
  curl -LsSf https://astral.sh/uv/install.sh | sh
fi
export PATH="$HOME/.local/bin:$PATH"
uv --version

echo "==> [3/5] code -> $CODE_DIR (branch=$BRANCH)"
if [ -d "$CODE_DIR/.git" ]; then
  git -C "$CODE_DIR" fetch --all --prune
  git -C "$CODE_DIR" checkout "$BRANCH"
  git -C "$CODE_DIR" pull --ff-only origin "$BRANCH" || true
else
  git clone --branch "$BRANCH" "$REPO_URL" "$CODE_DIR"
fi

echo "==> [4/5] python deps (uv sync --extra dev --extra logger)"
cd "$CODE_DIR"
uv sync --extra dev --extra logger
uv run python -c "import torch; print('torch', torch.__version__, 'cuda_available', torch.cuda.is_available())"

echo "==> [5/5] KITTI dataset -> $DATA_DIR (mode=$KITTI_MODE)"
DATA_DIR="$DATA_DIR" KITTI_MODE="$KITTI_MODE" bash "$CODE_DIR/deploy/vast/download_kitti.sh"

cat <<EOF

==> provisioning complete. Nothing has been trained yet.

Start training manually with, e.g.:
  cd $CODE_DIR
  DATA_DIR=$DATA_DIR CODE_DIR=$CODE_DIR bash deploy/vast/train_vast.sh

or directly:
  cd $CODE_DIR && uv run python train.py data_dir=$DATA_DIR trainer.accelerator=auto trainer.devices=1
EOF
