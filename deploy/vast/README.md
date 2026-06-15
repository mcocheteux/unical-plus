# Training UniCal on vast.ai

This directory contains everything needed to train UniCal on a rented GPU box on
[vast.ai](https://vast.ai), using the KITTI raw dataset.

> These are **preparation files + commands only**. Nothing here rents a machine
> or starts training on its own — you run the commands yourself when ready.

## Files

| File | Purpose |
|---|---|
| `provision.sh` | One-shot setup on the instance: system deps → `uv` → clone repo → `uv sync` → download KITTI. Does **not** train. Usable as the vast.ai `--onstart` script or run by hand. |
| `download_kitti.sh` | Downloads exactly the KITTI raw drives referenced by `configs/experiment/*.yaml` (idempotent). |
| `train_vast.sh` | Thin wrapper around `uv run python train.py` with GPU-friendly defaults. Run manually. |

## 0. Local prerequisites (your machine)

```bash
# Install the vast.ai CLI (any of these)
uv tool install vastai            # or: pipx install vastai / pip install vastai

# Authenticate. The key is provided to this project as the secret VAST_API_KEY.
vastai set api-key "$VAST_API_KEY"

# Confirm the key works (should print your account / email):
vastai show user
```

## 1. See what GPUs are available (read-only)

```bash
# Cheapest reliable single-GPU boxes with >=16 GB VRAM and >=80 GB disk,
# ordered by price. (KITTI default split + torch/CUDA wheels need ~60-80 GB.)
vastai search offers \
  'reliability>0.98 num_gpus=1 gpu_ram>=16 disk_space>=80 inet_down>=200 rentable=true verified=true' \
  -o 'dph_total' --limit 25

# Want a specific GPU? add e.g.  gpu_name=RTX_4090   (spaces -> underscores)
# JSON output for scripting:     add  --raw
```

Note the offer **ID** in the first column — you pass it to `create instance`.

## 2. Rent the instance (you run this when ready)

```bash
OFFER_ID=<id-from-step-1>

vastai create instance "$OFFER_ID" \
  --image pytorch/pytorch:2.4.0-cuda12.1-cudnn9-devel \
  --disk 100 \
  --ssh --direct \
  --label unical-kitti \
  --onstart deploy/vast/provision.sh \
  --env '-e KITTI_MODE=default -e BRANCH=main'
```

- `--onstart deploy/vast/provision.sh` runs provisioning automatically on boot
  (clone + deps + KITTI download), but **does not** start training.
- Prefer to do it by hand instead? Drop `--onstart` and run `provision.sh`
  yourself after step 3.
- For a quick end-to-end smoke test on a tiny dataset, use
  `--env '-e KITTI_MODE=debug'` and a smaller `--disk 60`.

Check status / get the SSH endpoint:

```bash
vastai show instances
INSTANCE_ID=<id-from-show-instances>
vastai ssh-url "$INSTANCE_ID"      # ssh://root@host:port
vastai logs "$INSTANCE_ID"          # onstart / provisioning logs
```

## 3. Connect and (if needed) provision

```bash
ssh -p <port> root@<host>           # from `vastai ssh-url`

# If you did NOT use --onstart, run provisioning now (idempotent):
cd /workspace/unical 2>/dev/null || \
  (git clone -b main https://github.com/mcocheteux/unical.git /workspace/unical && cd /workspace/unical)
KITTI_MODE=default bash deploy/vast/provision.sh
```

## 4. Train (you run this when ready)

```bash
cd /workspace/unical

# GPU-friendly defaults via the wrapper (forwards Hydra overrides):
bash deploy/vast/train_vast.sh trainer.max_epochs=500 data.batch_size=16

# …or call train.py directly:
uv run python train.py \
  data_dir=/workspace/kitti_raw \
  trainer.accelerator=auto trainer.devices=1 \
  data.num_workers=8 data.batch_size=16

# Quick smoke test (needs KITTI_MODE=debug data):
uv run python train.py data_dir=/workspace/kitti_raw experiment=debug trainer.max_epochs=1
```

Tip: run long jobs under `tmux`/`screen` so they survive the SSH session, and
watch GPU usage with `nvidia-smi -l 5`.

## 5. Retrieve checkpoints / logs

Checkpoints land in `/workspace/unical/logs/checkpoints/` (git-ignored on the box).

```bash
# from your local machine
vastai scp-url "$INSTANCE_ID"       # prints an scp endpoint
scp -P <port> -r root@<host>:/workspace/unical/logs ./logs-from-vast
```

## 6. Stop / destroy (avoid idle charges)

```bash
vastai stop instance "$INSTANCE_ID"      # keeps disk, billable storage only
vastai destroy instance "$INSTANCE_ID"   # irreversible: deletes the box + data
```

## Dataset notes

`download_kitti.sh` fetches:

- **default** (`configs/experiment/default.yaml`): `2011_09_26` drives
  `1,2,9,11,13,14,15,17,18,19,20,22,23,27,28,29,32,35,36,39,46,48,51,52,56,57,59,60,61,64,79,84,86,87,91,93,95,96,101,104,106,113,117` (train) + `5,70` (val), and `2011_09_30` drive `28` (test), plus both calibration sets. Expect tens of GB.
- **debug** (`configs/experiment/debug.yaml`): only `2011_09_26` drives `1` and `5`.

KITTI raw zips unpack to `<DATA_DIR>/<date>/<date>_drive_NNNN_sync/...` and
`<DATA_DIR>/<date>/calib_*.txt`, which is exactly the layout
`unical.data.dataset.KittiDataset` reads, so just pass `data_dir=<DATA_DIR>`.
