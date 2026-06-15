#!/usr/bin/env bash
#
# Convenience wrapper to launch UniCal training on a (GPU) vast.ai instance.
# Run manually after provisioning. Any extra args are forwarded to train.py as
# Hydra overrides, e.g.:
#
#   bash deploy/vast/train_vast.sh trainer.max_epochs=200 data.batch_size=16
#   bash deploy/vast/train_vast.sh experiment=debug          # quick smoke train
#
set -euo pipefail

CODE_DIR="${CODE_DIR:-/workspace/unical}"
DATA_DIR="${DATA_DIR:-/workspace/kitti_raw}"
export PATH="$HOME/.local/bin:$PATH"

cd "$CODE_DIR"

# Weights & Biases tracking (charts + best/last checkpoints). Requires the
# `logger` extra (provision.sh installs it) and WANDB_API_KEY in the env
# (or run `wandb login`). Pass `logger=csv` to disable.
if [ -z "${WANDB_API_KEY:-}" ]; then
  echo "[train] WARNING: WANDB_API_KEY is not set — W&B will not be able to sync."
  echo "[train]          export WANDB_API_KEY=... (or run 'wandb login') first,"
  echo "[train]          or pass 'logger=csv' to log locally only."
fi

# accelerator=auto picks CUDA on a GPU box; bump num_workers for the bigger machine.
exec uv run python train.py \
  data_dir="$DATA_DIR" \
  trainer.accelerator=auto \
  trainer.devices=1 \
  data.num_workers=8 \
  logger=wandb \
  "$@"
