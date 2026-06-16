# UniCal — Camera-LiDAR Extrinsic Calibration

**UniCal** is a deep learning model that estimates the rigid-body correction
between a camera and a LiDAR sensor from a single paired image and point-cloud
scan.  Accurate extrinsic calibration is essential for sensor fusion in
autonomous driving, but factory calibrations drift over time.  UniCal learns to
regress the correction end-to-end without a calibration target.

Built with PyTorch Lightning and Hydra for reproducibility and clean
configuration management.

---

## Architecture

```
RGB image  (H × W × 3) ─┐
                          ├─► channel concat ─► MobileViT-S backbone
LiDAR map  (H × W × 1) ─┘         (early fusion, 4-channel input)
                                          │
                               global average pool
                                          │
                                   640-dim feature
                                          │
                              SplitRegressionHead
                               /                \
                        trans (3)           rot-6D (6)
                                                 │
                                      Gram-Schmidt → SO(3)
                                         rotation matrix
```

**Key design choices:**

- **Early fusion** — the RGB image and the LiDAR inverse-depth map are
  concatenated channel-wise before entering a single backbone.  A
  MobileViT-Small pretrained on ImageNet is used; its 3-channel convolutional
  stem is inflated to 4 channels (extra channel initialised to the mean of the
  RGB filters) so the pretrained representations transfer with minimal
  disruption.

- **6-D continuous rotation** — the head predicts a 6-D vector that is
  mapped to SO(3) via Gram-Schmidt orthonormalisation (Zhou et al., CVPR 2019).
  This avoids the discontinuities of Euler angles and the double-cover of
  quaternions, making gradient-based optimisation well-conditioned.

- **Combined loss** — a regression term (MSE on translation + Frobenius MSE on
  rotation matrix) is summed with a spatial term that measures the discrepancy
  between ground-truth and predicted point-cloud projections (centroid distance
  + per-point distance), inspired by LCCNet (Lv et al., CVPR 2021).

---

## Project Structure

```
unical/
├── train.py                  # Training entry point (Hydra CLI)
├── evaluate.py               # Standalone test from checkpoint (Hydra CLI)
├── configs/
│   ├── train.yaml            # Root config: trainer, paths, seed
│   ├── data/kitti.yaml       # Dataset, preprocessor, decalibrator
│   ├── model/unical.yaml     # Backbone, head, loss architecture
│   ├── experiment/
│   │   ├── default.yaml      # Full KITTI split (LCCNet convention)
│   │   └── debug.yaml        # Tiny split for smoke tests
│   └── logger/
│       ├── csv.yaml          # Default: CSV logs
│       └── wandb.yaml        # Optional: Weights & Biases
├── src/unical/
│   ├── models/
│   │   ├── backbone.py       # MobileViT early-fusion backbone
│   │   ├── head.py           # Split regression head (trans + rot)
│   │   └── module.py         # UniCal LightningModule
│   ├── losses/
│   │   ├── regression.py     # MSE on translation + rotation matrix
│   │   ├── spatial.py        # Point-cloud projection loss
│   │   └── combined.py       # Weighted sum of both loss terms
│   ├── data/
│   │   ├── dataset.py        # KittiDataset, Batch
│   │   ├── datamodule.py     # KittiDataModule (Lightning)
│   │   ├── preprocessor.py   # Image resize, LiDAR projection, augmentation
│   │   └── decalibrator.py   # Random rigid-body error generator
│   └── utils/
│       ├── transform.py      # Transform class, rotation helpers
│       ├── geometry.py       # Point-cloud projection, normalisation
│       ├── augmentation.py   # Photometric, point-cloud, spatial distortion
│       └── metrics.py        # Translation/rotation error metrics
├── tests/                    # pytest suite
└── deploy/vast/              # Scripts for Vast.ai GPU cloud setup
```

---

## Quick Start

### 1. Install

```bash
git clone https://github.com/mcocheteux/unical.git
cd unical

# with uv (recommended)
uv sync --extra dev

# or with pip
pip install -e ".[dev]"
```

### 2. Prepare the KITTI Raw dataset

Download the KITTI Raw data from
[cvlibs.net](https://www.cvlibs.net/datasets/kitti/raw_data.php).
The expected directory layout is:

```
<data_dir>/
└── 2011_09_26/
    ├── calib_cam_to_cam.txt
    ├── calib_velo_to_cam.txt
    ├── 2011_09_26_drive_0001_sync/
    │   ├── image_02/data/*.png
    │   └── velodyne_points/data/*.bin
    └── ...
└── 2011_09_30/
    └── 2011_09_30_drive_0028_sync/
        └── ...
```

The `deploy/vast/` scripts automate downloading and training on a
[Vast.ai](https://vast.ai) GPU instance.

### 3. Train

```bash
# default config (full split, CSV logging)
python train.py data_dir=/path/to/kitti_raw

# with Weights & Biases logging
python train.py data_dir=/path/to/kitti_raw logger=wandb

# fast debug run on CPU (tiny split, 2 epochs)
python train.py data_dir=/path/to/kitti_raw experiment=debug \
    trainer.max_epochs=2 trainer.accelerator=cpu data.num_workers=0

# mixed-precision on Ampere+ GPU (~40 % faster, less memory)
python train.py data_dir=/path/to/kitti_raw trainer.precision=bf16-mixed
```

### 4. Evaluate a checkpoint

```bash
python evaluate.py data_dir=/path/to/kitti_raw +ckpt=logs/checkpoints/last.ckpt
```

---

## Configuration System

UniCal uses [Hydra](https://hydra.cc) for configuration.  All values in the
YAML files can be overridden on the command line with `key=value` syntax.

| Override | Default | Description |
|---|---|---|
| `data_dir` | *(required)* | Path to KITTI raw root directory |
| `experiment` | `default` | Use `debug` for a quick smoke test |
| `logger` | `csv` | Set to `wandb` for Weights & Biases tracking |
| `trainer.max_epochs` | `500` | Number of training epochs |
| `trainer.precision` | `32-true` | `bf16-mixed` recommended on Ampere+ GPUs |
| `trainer.devices` | `1` | Number of GPUs |
| `trainer.accelerator` | `auto` | Auto-selects CUDA / MPS / CPU |
| `model.backbone.pretrained` | `apple/mobilevit-small` | Set to `null` to train from scratch |
| `data.batch_size` | `8` | Samples per GPU |
| `data.num_workers` | `4` | DataLoader workers (set to `0` on macOS if needed) |

Config files live in `configs/`.  See `configs/model/unical.yaml` for full
architecture parameters and `configs/data/kitti.yaml` for dataset / augmentation
settings.

---

## Metrics

The model is evaluated on the held-out test sequence (`2011_09_30` drive 28,
following the LCCNet split).  Reported metrics:

- **Translation error** — mean absolute error (MAE) and standard deviation in
  centimetres, computed per axis and overall.
- **Rotation error** — geodesic angle error in degrees between predicted and
  ground-truth rotation matrices.

Decalibration errors are sampled uniformly in [−1°, +1°] (rotation) and
[−10 cm, +10 cm] (translation) per axis, matching the LCCNet training regime.

---

## References

- **MobileViT**: Mehta & Rastegari, *MobileViT: Light-weight, General-purpose,
  and Mobile-friendly Vision Transformer*, ICLR 2022.
  [`apple/mobilevit-small`](https://huggingface.co/apple/mobilevit-small)

- **6-D rotation representation**: Zhou et al., *On the Continuity of Rotation
  Representations in Neural Networks*, CVPR 2019.

- **Spatial loss / data split**: Lv et al., *LCCNet: LiDAR and Camera
  Self-Calibration using Cost Volume*, CVPR 2021.

- **Dataset**: Geiger et al., *Are we ready for Autonomous Driving? The KITTI
  Vision Benchmark Suite*, CVPR 2012.

---

## License

MIT — see [LICENSE](LICENSE).
