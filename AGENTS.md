# AGENTS.md

## Cursor Cloud specific instructions

UniCal is a single Python package (`src/unical`), not a service — there is **no
server, database, or web UI to run**. The "application" is two Hydra CLI entry
points: `train.py` (train + test) and `evaluate.py` (test from a checkpoint).
It trains a MobileViT-based camera-LiDAR extrinsic-calibration model.

### Tooling
- Package manager is **uv** (lockfile `uv.lock`, deps in `pyproject.toml`,
  `requires-python >=3.11`). `uv` lives in `~/.local/bin` and is already on PATH
  for login shells (the installer added it to `~/.bashrc`/`~/.profile`).
- Dependencies are installed by the startup update script (`uv sync --extra dev`),
  so they are ready on session start. Run commands with `uv run ...`.

### Lint / test (standard, see `pyproject.toml`)
- Lint: `uv run ruff check .` — note the repo currently has **pre-existing**
  ruff findings; a non-zero exit does not mean your environment is broken.
- Tests: `uv run pytest` — runs the suite in `tests/` (model, losses, transforms,
  augmentation, geometry). All tests use synthetic tensors; no KITTI data needed.

### Running the app (non-obvious caveats)
- Both `train.py` and `evaluate.py` **require a KITTI-raw dataset on disk** via
  `data_dir=...` (a mandatory Hydra field). The repo ships **no data**.
- This VM is **CPU-only** (`torch.cuda.is_available()` is `False`). The config
  default `trainer.accelerator=auto` resolves to CPU fine; pass
  `data.num_workers=0` to avoid DataLoader worker issues on quick runs.
- `experiment=debug` selects tiny splits (drive 1 = train, drive 5 = val/test).
- `evaluate.py` needs the checkpoint added with a leading `+`
  (`+ckpt=/path/to.ckpt`) because Hydra runs in struct mode; plain `ckpt=` errors.
- Outputs (checkpoints, logs) go to `logs/` (git-ignored).

### Smoke-testing without the real (multi-GB) KITTI dataset
`unical.data.dataset.KittiDataset` only reads files off disk, so a tiny synthetic
tree in the KITTI-raw layout is enough to exercise the full pipeline end-to-end:
```
<data_dir>/2011_09_26/calib_cam_to_cam.txt          # line "P_rect_02: <12 floats>"
<data_dir>/2011_09_26/calib_velo_to_cam.txt         # lines "R: <9 floats>" and "T: <3 floats>"
<data_dir>/2011_09_26/2011_09_26_drive_0001_sync/image_02/data/<10-digit>.png
<data_dir>/2011_09_26/2011_09_26_drive_0001_sync/velodyne_points/data/<10-digit>.bin   # float32 (N,4)
# plus drive 0005 for the debug val/test split
```
Then: `uv run python train.py data_dir=<data_dir> experiment=debug trainer.max_epochs=1 trainer.accelerator=cpu data.batch_size=2 data.num_workers=0`
