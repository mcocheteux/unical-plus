<div align="center">

# UniCal

### Camera-LiDAR Extrinsic Calibration via Transformer-Based Early Fusion

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-ee4c2c.svg)](https://pytorch.org/)
[![Lightning](https://img.shields.io/badge/Lightning-2.2+-792ee5.svg)](https://lightning.ai/)
[![License: CC BY-NC 4.0](https://img.shields.io/badge/License-CC%20BY--NC%204.0-lightgrey.svg)](LICENSE)

*Unofficial reimplementation — see [Attribution](#attribution) below.*

</div>

---

Accurate extrinsic calibration between cameras and LiDAR sensors is critical for
autonomous vehicle perception. Factory calibrations degrade over time, and
target-based re-calibration is expensive.

**UniCal** learns to estimate the correction from a single snapshot — one RGB
image + one LiDAR scan — with no calibration target needed. It processes both
modalities through a shared MobileViT backbone (early fusion), predicts a 6-D
continuous rotation and a translation vector, and is supervised by a combined
regression + 3-D spatial loss.

---

## Highlights

| | |
|---|---|
| 🔀 **Early fusion** | Camera image and LiDAR depth map are channel-concatenated before the backbone — a single model sees both modalities from the first layer |
| 🔄 **6-D rotation** | Continuous SO(3) representation (Zhou et al., CVPR 2019) avoids gimbal lock and quaternion double-cover |
| 📐 **Spatial loss** | Differentiable point-cloud projection loss penalises 3-D misalignment directly, not just regression targets |
| ⚡ **AMP-safe geometry** | Rigid-transform math is pinned to full precision inside mixed-precision training |
| 🧩 **Hydra config** | Every hyperparameter is a CLI override — no code edits needed to run experiments |
| 🔬 **46 tests** | Full pytest suite covering models, losses, transforms, augmentation, and geometry |

---

## Architecture

```
 Camera image   (H × W × 3) ──┐
                                ├──► 4-channel concat
 LiDAR inv-depth (H × W × 1) ─┘         │
                                          ▼
                               MobileViT-S backbone
                               (pretrained, stem inflated
                                3 ch → 4 ch at init)
                                          │
                               Global average pool
                                          │
                                  640-dim feature vector
                                          │
                            ┌─────────────┴─────────────┐
                            ▼                           ▼
                    Translation head             Rotation head
                       MLP → (3,)               MLP → (6,)
                                                     │
                                          Gram-Schmidt → SO(3)
                                             (3 × 3 matrix)
```

The predicted decalibration is inverted analytically (R⁻¹ = Rᵀ, t⁻¹ = −Rᵀt)
and composed with the noisy initial extrinsic to produce the recalibrated
transform — no matrix inverse needed, gradients flow cleanly.

---

## Quick Start

### Install

```bash
git clone https://github.com/mcocheteux/unical.git
cd unical

# recommended
uv sync --extra dev

# or
pip install -e ".[dev]"
```

### Prepare KITTI Raw

Download from [cvlibs.net](https://www.cvlibs.net/datasets/kitti/raw_data.php).
Expected layout:

```
<data_dir>/
├── 2011_09_26/
│   ├── calib_cam_to_cam.txt
│   ├── calib_velo_to_cam.txt
│   └── 2011_09_26_drive_XXXX_sync/
│       ├── image_02/data/*.png
│       └── velodyne_points/data/*.bin
└── 2011_09_30/
    └── 2011_09_30_drive_0028_sync/   ← test sequence
```

> The `deploy/vast/` scripts automate provisioning, data download, and training
> on a [Vast.ai](https://vast.ai) GPU instance.

### Train

```bash
# Full run
python train.py data_dir=/path/to/kitti_raw

# W&B logging
python train.py data_dir=/path/to/kitti_raw logger=wandb

# Ampere+ GPU (bf16, ~40 % faster)
python train.py data_dir=/path/to/kitti_raw trainer.precision=bf16-mixed

# CPU smoke test (tiny split, 2 epochs)
python train.py data_dir=/path/to/kitti_raw experiment=debug \
    trainer.max_epochs=2 trainer.accelerator=cpu data.num_workers=0
```

### Evaluate

```bash
python evaluate.py data_dir=/path/to/kitti_raw +ckpt=logs/checkpoints/last.ckpt
```

### Run tests

```bash
uv run pytest -v
```

---

## Configuration

UniCal uses [Hydra](https://hydra.cc) — every YAML value is a CLI override.

| Key | Default | Notes |
|---|---|---|
| `data_dir` | **required** | Path to KITTI raw root |
| `experiment` | `default` | `debug` = tiny split for fast iteration |
| `logger` | `csv` | `wandb` for W&B experiment tracking |
| `trainer.precision` | `32-true` | `bf16-mixed` on Ampere+ GPUs |
| `trainer.max_epochs` | `500` | |
| `trainer.devices` | `1` | |
| `model.backbone.pretrained` | `apple/mobilevit-small` | `null` = train from scratch |
| `data.batch_size` | `8` | Per GPU |
| `data.num_workers` | `4` | Set to `0` on macOS if workers crash |

See [`configs/model/unical.yaml`](configs/model/unical.yaml) for full
architecture parameters and [`configs/data/kitti.yaml`](configs/data/kitti.yaml)
for dataset and augmentation settings.

---

## Evaluation Protocol

Tested on KITTI Raw `2011_09_30` drive 28 (LCCNet/RegNet split).
Decalibration errors sampled uniformly in **±1°** (rotation) and **±10 cm**
(translation) per axis.

**Metrics:**

- Translation MAE and σ in centimetres (per axis + overall)
- Geodesic rotation error in degrees

---

## Project Structure

```
unical/
├── train.py                    # Training entry point
├── evaluate.py                 # Inference from checkpoint
├── configs/
│   ├── train.yaml              # Root config
│   ├── model/unical.yaml       # Architecture + loss
│   ├── data/kitti.yaml         # Dataset + augmentation
│   └── experiment/             # Train/val/test splits
├── src/unical/
│   ├── models/                 # Backbone, head, LightningModule
│   ├── losses/                 # Regression, spatial, combined
│   ├── data/                   # Dataset, DataModule, preprocessor
│   └── utils/                  # Transform, geometry, augmentation, metrics
├── tests/                      # 46 pytest tests (no data required)
└── deploy/vast/                # Vast.ai GPU provisioning scripts
```

---

## Attribution

This is an **unofficial reimplementation** of:

> Cocheteux, Low & Bruehlmeier, *"UniCal: a Single-Branch Transformer-Based
> Model for Camera-to-LiDAR Calibration and Validation"*, arXiv:2304.09715, 2023.
> <https://arxiv.org/abs/2304.09715>

The implementation incorporates independent modifications. It is not affiliated
with or endorsed by the original institution, and no proprietary code is
included.

---

## References

- Cocheteux et al., *UniCal* — arXiv:2304.09715, 2023 *(paper this reimplements)*
- Mehta & Rastegari, *MobileViT* — ICLR 2022 · [`apple/mobilevit-small`](https://huggingface.co/apple/mobilevit-small)
- Zhou et al., *On the Continuity of Rotation Representations* — CVPR 2019
- Lv et al., *LCCNet* — CVPR 2021 *(spatial loss & data split)*
- Geiger et al., *KITTI* — CVPR 2012

---

## License

[CC BY-NC 4.0](LICENSE) — free for research and personal use; **commercial use is not permitted.**
