"""
End-to-end smoke test for the KITTI-C2L adapter: builds a synthetic KITTI raw
drive + a real KITTI-C2L parquet file (via the kitti_c2l package), then runs
one batch all the way through the real UniCal model on CPU (pretrained=None,
so no network access is needed). This is the "verify on a tiny slice without
GPU or real data" check for the benchmark pipeline.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from kitti_c2l.generate import generate_sequence_rows
from kitti_c2l.sequence_mapping import SequenceInfo
from kitti_c2l.stages import STAGES
from kitti_c2l.writer import write_split
from PIL import Image

from unical.data.kitti_c2l_dataset import KittiC2LDataModule, KittiC2LDataset
from unical.data.preprocessor import DataPreprocessor, PreprocessorConfig
from unical.losses.combined import CombinedLoss
from unical.losses.regression import RegressionLoss
from unical.losses.spatial import SpatialLoss
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead
from unical.models.module import UniCal

IMAGE_SIZE = 64  # (width, height) of the synthetic fixture -- keep tiny & fast


def _make_fake_kitti_date(date_dir: Path, drive: str, num_frames: int) -> None:
    drive_dir = date_dir / f"{date_dir.name}_drive_{drive}_sync"
    image_dir = drive_dir / "image_02" / "data"
    lidar_dir = drive_dir / "velodyne_points" / "data"
    image_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(0)
    for frame in range(num_frames):
        image = rng.integers(0, 255, size=(IMAGE_SIZE, IMAGE_SIZE, 3), dtype=np.uint8)
        Image.fromarray(image).save(image_dir / f"{frame:010d}.png")

        num_points = 64
        points = rng.uniform(-5.0, 5.0, size=(num_points, 3)).astype(np.float32)
        points[:, 2] = rng.uniform(2.0, 20.0, size=num_points).astype(np.float32)
        reflectance = rng.uniform(0.0, 1.0, size=(num_points, 1)).astype(np.float32)
        np.concatenate([points, reflectance], axis=1).tofile(lidar_dir / f"{frame:010d}.bin")

    (date_dir / "calib_imu_to_velo.txt").write_text(
        "calib_time: 15-Mar-2012 11:37:16\nR: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\nT: 0.8 0.0 0.3\n"
    )
    (date_dir / "calib_velo_to_cam.txt").write_text(
        "calib_time: 15-Mar-2012 11:37:16\n"
        "R: 7.533745e-03 -9.999714e-01 -6.166020e-04 "
        "1.480249e-02 7.280733e-04 -9.998902e-01 "
        "9.998621e-01 7.523790e-03 1.480755e-02\n"
        "T: -4.069766e-03 -7.631618e-02 -2.717806e-01\n"
    )
    focal_length = 700.0
    p_rect = f"{focal_length} 0.0 {IMAGE_SIZE / 2} -40.0 0.0 {focal_length} {IMAGE_SIZE / 2} 0.0 0.0 0.0 1.0 0.0"
    (date_dir / "calib_cam_to_cam.txt").write_text(
        "calib_time: 15-Mar-2012 11:37:16\n"
        "R_rect_00: 1.0 0.0 0.0 0.0 1.0 0.0 0.0 0.0 1.0\n"
        f"S_rect_00: {float(IMAGE_SIZE)} {float(IMAGE_SIZE)}\n"
        f"P_rect_00: {p_rect}\n"
        f"S_rect_02: {float(IMAGE_SIZE)} {float(IMAGE_SIZE)}\n"
        f"P_rect_02: {p_rect}\n"
    )


@pytest.fixture
def kitti_c2l_fixture(tmp_path: Path) -> tuple[Path, Path]:
    """Builds a synthetic KITTI raw root + KITTI-C2L train/test parquet pair.

    Two fake sequences ("00" train, "09" test), each backed by its own
    fake raw drive, 6 frames each.
    """
    kitti_raw_root = tmp_path / "kitti_raw"
    _make_fake_kitti_date(kitti_raw_root / "2011_10_03", drive="0027", num_frames=6)
    _make_fake_kitti_date(kitti_raw_root / "2011_09_30", drive="0033", num_frames=6)

    train_seq = SequenceInfo(
        sequence="00", date="2011_10_03", drive="0027", start=0, end=5, frame_count=6, split="train"
    )
    test_seq = SequenceInfo(
        sequence="09", date="2011_09_30", drive="0033", start=0, end=5, frame_count=6, split="test"
    )

    data_dir = tmp_path / "data"
    write_split(
        generate_sequence_rows(train_seq, kitti_raw_root, stages=STAGES[:2]),
        data_dir / "train.parquet",
    )
    write_split(
        generate_sequence_rows(test_seq, kitti_raw_root, stages=STAGES[:2]),
        data_dir / "test.parquet",
    )

    return kitti_raw_root, data_dir


def _small_preprocessor() -> DataPreprocessor:
    return DataPreprocessor(PreprocessorConfig(width=IMAGE_SIZE, height=IMAGE_SIZE))


def test_dataset_getitem_shapes(kitti_c2l_fixture: tuple[Path, Path]):
    kitti_raw_root, data_dir = kitti_c2l_fixture
    dataset = KittiC2LDataset(data_dir / "train.parquet", kitti_raw_root, _small_preprocessor())
    assert len(dataset) == 6 * 2  # 6 frames x 2 stages

    sample = dataset[0]
    assert sample["img"].shape == (1, 3, IMAGE_SIZE, IMAGE_SIZE)
    assert sample["lidar_map"].shape[2:] == (IMAGE_SIZE, IMAGE_SIZE)
    assert sample["trans"].shape == (3,)
    assert sample["rot_mat"].shape == (3, 3)
    assert sample["metadata"][0]["sequence"] == "00"


def test_decal_target_reconstructs_miscalibration(kitti_c2l_fixture: tuple[Path, Path]):
    """T_decal @ T_gt must reconstruct T_init (== the published miscalibrated extrinsic)."""
    kitti_raw_root, data_dir = kitti_c2l_fixture
    dataset = KittiC2LDataset(data_dir / "train.parquet", kitti_raw_root, _small_preprocessor())
    sample = dataset[3]
    meta = sample["metadata"][0]
    reconstructed = meta["T_decal"] @ meta["T_gt"]
    np.testing.assert_allclose(reconstructed.matrix, meta["T_init"].matrix, atol=1e-4)


def test_datamodule_val_sequence_excluded_from_train(kitti_c2l_fixture: tuple[Path, Path]):
    kitti_raw_root, data_dir = kitti_c2l_fixture
    dm = KittiC2LDataModule(
        data_dir=str(data_dir),
        kitti_raw_root=str(kitti_raw_root),
        preprocessor=_small_preprocessor(),
        val_sequence="00",  # our only train sequence -- so train_ds should end up empty
        batch_size=2,
        num_workers=0,
        pin_memory=False,
    )
    dm.setup()
    assert len(dm.train_ds) == 0
    assert len(dm.val_ds) == 6 * 2
    assert len(dm.test_ds) == 6 * 2


@pytest.mark.parametrize("num_workers", [0, 2])
def test_full_pipeline_through_real_model_cpu(kitti_c2l_fixture: tuple[Path, Path], num_workers):
    """One batch, real KittiC2LDataModule -> real UniCal model -> real loss, on CPU."""
    kitti_raw_root, data_dir = kitti_c2l_fixture
    dm = KittiC2LDataModule(
        data_dir=str(data_dir),
        kitti_raw_root=str(kitti_raw_root),
        preprocessor=_small_preprocessor(),
        val_sequence="nonexistent",  # keep all 6 train frames
        batch_size=2,
        num_workers=num_workers,
        pin_memory=False,
    )
    dm.setup()
    batch = next(iter(dm.train_dataloader()))

    backbone = MobileViTBackbone(image_size=IMAGE_SIZE, pretrained=None, spatial_head=True)
    feat_dim = backbone.model.config.neck_hidden_sizes[-1]
    head = SplitRegressionHead(
        in_features=feat_dim, common_hidden=[], trans_hidden=[32], rot_hidden=[32], rot_dim=6
    )
    loss = CombinedLoss(
        regression=RegressionLoss(trans_weight=1.0, rot_weight=1.0),
        spatial=SpatialLoss(centroid_weight=1.0, pcl_weight=1.0),
    )
    model = UniCal(backbone=backbone, head=head, loss=loss, lr=1e-4)

    with torch.no_grad():
        losses, pred_Ts, target_Ts = model._step(batch)
    assert torch.isfinite(losses["loss"])
    assert len(pred_Ts) == len(target_Ts) == batch.img.shape[0]
