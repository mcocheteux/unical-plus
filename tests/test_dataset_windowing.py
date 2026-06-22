"""Tests for KittiDataset windowing (sequence_length / frame_stride)."""

from pathlib import Path

import cv2
import numpy as np
import pytest

from unical.data.dataset import KittiDataset
from unical.data.decalibrator import DualErrorGenerator
from unical.data.preprocessor import DataPreprocessor, PreprocessorConfig

DATE = "2011_09_26"
DRIVE = 1


def _write_calib(date_dir: Path) -> None:
    date_dir.mkdir(parents=True, exist_ok=True)
    (date_dir / "calib_cam_to_cam.txt").write_text("P_rect_02: 700 0 320 0 0 700 240 0 0 0 1 0\n")
    (date_dir / "calib_velo_to_cam.txt").write_text("R: 1 0 0 0 1 0 0 0 1\nT: 0 0 0\n")


def _write_frame(drive_dir: Path, fid: int) -> None:
    img_dir = drive_dir / "image_02" / "data"
    lidar_dir = drive_dir / "velodyne_points" / "data"
    img_dir.mkdir(parents=True, exist_ok=True)
    lidar_dir.mkdir(parents=True, exist_ok=True)

    img = np.full((64, 64, 3), 128, dtype=np.uint8)
    cv2.imwrite(str(img_dir / f"{fid:010d}.png"), img)

    pcl = np.zeros((10, 4), dtype=np.float32)
    pcl[:, 0] = 5.0  # 5 m in front, within [min_depth, max_depth]
    pcl[:, 3] = 1.0  # intensity > 0
    pcl.tofile(str(lidar_dir / f"{fid:010d}.bin"))


def _make_drive(data_dir: Path, n_frames: int, missing: set[int] | None = None) -> None:
    missing = missing or set()
    date_dir = data_dir / DATE
    _write_calib(date_dir)
    drive_dir = date_dir / f"{DATE}_drive_{DRIVE:04d}_sync"
    for fid in range(n_frames):
        if fid in missing:
            continue
        _write_frame(drive_dir, fid)


def _dataset(data_dir: Path, sequence_length: int = 1, frame_stride: int = 1) -> KittiDataset:
    preprocessor = DataPreprocessor(PreprocessorConfig(width=32, height=32))
    decalibrator = DualErrorGenerator(r_range_lidar=1.0, t_range_lidar=10.0)
    return KittiDataset(
        data_dir=data_dir,
        split=[(DATE, [DRIVE])],
        preprocessor=preprocessor,
        decalibrator=decalibrator,
        sequence_length=sequence_length,
        frame_stride=frame_stride,
        deterministic=True,
    )


def test_sequence_length_one_matches_frame_count(tmp_path: Path):
    _make_drive(tmp_path, n_frames=5)
    ds = _dataset(tmp_path, sequence_length=1, frame_stride=1)
    assert len(ds) == 5


def test_window_count_for_sequence_length_and_stride(tmp_path: Path):
    _make_drive(tmp_path, n_frames=10)
    ds = _dataset(tmp_path, sequence_length=3, frame_stride=1)
    # windows starting at fids 0..7 (need fid, fid+1, fid+2 to exist)
    assert len(ds) == 8

    ds_stride2 = _dataset(tmp_path, sequence_length=3, frame_stride=2)
    # span = (3-1)*2 = 4, windows starting at fids 0..5
    assert len(ds_stride2) == 6


def test_dropped_window_on_gap(tmp_path: Path):
    _make_drive(tmp_path, n_frames=10, missing={5})
    ds = _dataset(tmp_path, sequence_length=3, frame_stride=1)
    # any window spanning fid 5 is dropped: windows starting 3,4,5 (cover 3-5,4-6,5-7)
    # valid_fids excludes 5, so contiguity check drops windows whose ids aren't
    # all present and exactly contiguous.
    fids_used = []
    for date, drive, window in ds._samples:
        fids_used.append(window)
    assert all(5 not in w for w in fids_used)


def test_getitem_window_shapes(tmp_path: Path):
    _make_drive(tmp_path, n_frames=5)
    ds = _dataset(tmp_path, sequence_length=3, frame_stride=1)
    sample = ds[0]
    assert sample["img"].shape == (3, 3, 32, 32)
    assert sample["lidar_map"].shape == (3, 1, 32, 32)
    assert len(sample["pcl"]) == 3
    assert len(sample["metadata"]) == 3
    # decalibration is shared across the whole window
    T_decal_0 = sample["metadata"][0]["T_decal"]
    T_decal_1 = sample["metadata"][1]["T_decal"]
    np.testing.assert_allclose(T_decal_0.matrix, T_decal_1.matrix)


def test_collate_produces_per_frame_lists(tmp_path: Path):
    _make_drive(tmp_path, n_frames=5)
    ds = _dataset(tmp_path, sequence_length=2, frame_stride=1)
    batch = KittiDataset.collate([ds[0], ds[1]])
    assert batch.img.shape[:2] == (2, 2)  # (B, T, ...)
    assert len(batch.pcl) == 2  # one padded batch per window position
    assert batch.pcl[0].shape[0] == 2  # B
    assert len(batch.metadata) == 2
    assert len(batch.metadata[0]) == 2  # B per-sample dicts


def test_single_decalibration_draw_per_window(tmp_path: Path, monkeypatch):
    _make_drive(tmp_path, n_frames=5)
    ds = _dataset(tmp_path, sequence_length=3, frame_stride=1)

    call_count = 0
    original_call = DualErrorGenerator.__call__

    def counting_call(self, *args, **kwargs):
        nonlocal call_count
        call_count += 1
        return original_call(self, *args, **kwargs)

    monkeypatch.setattr(DualErrorGenerator, "__call__", counting_call)
    ds[0]
    assert call_count == 1


def test_invalid_sequence_length_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        _dataset(tmp_path, sequence_length=0)


def test_invalid_frame_stride_raises(tmp_path: Path):
    with pytest.raises(ValueError):
        _dataset(tmp_path, frame_stride=0)
