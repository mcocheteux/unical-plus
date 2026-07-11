"""
KITTI-C2L benchmark dataset for camera-LiDAR calibration.

Adapts the published KITTI-C2L-Dataset (github.com/mcocheteux/KITTI-C2L-Dataset)
-- fixed, seeded, per-(sequence, frame, stage) decalibrations -- into the same
``Batch`` contract as :class:`unical.data.dataset.KittiDataset`, so ``train.py``
/``evaluate.py`` work unchanged by switching ``data=kitti_c2l``.

KITTI-C2L publishes independent camera-mount and LiDAR-mount deltas (see its
README's "Decalibration protocol"), but UniCal -- like RegNet/CalibNet/LCCNet/
PseudoCal -- predicts a single *relative* LiDAR-to-camera correction. This
adapter derives that single target directly from the published ground-truth
and miscalibrated extrinsics: ``T_decal = T_miscal (.) T_gt^-1``, which is
exactly the transform that satisfies ``T_init = T_decal @ T_gt`` -- the same
relationship :class:`ErrorGenerator`-based training already assumes. This
works regardless of whether the underlying miscalibration came from a single
relative perturbation or two independent per-sensor ones.

Units: KITTI-C2L publishes translations in **centimetres**; this codebase's
``Transform`` (and ``DataPreprocessor``/``ErrorGenerator``) use **metres**
throughout -- converted here at the boundary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pyarrow.parquet as pq
import pytorch_lightning as L
import torch
from torch.utils.data import DataLoader, Dataset

from unical.data.dataset import Batch, KittiDataset
from unical.data.preprocessor import DataPreprocessor
from unical.utils.geometry import load_image_rgb
from unical.utils.transform import Transform

CM_TO_M = 1.0 / 100.0


def _drive_dir(kitti_raw_root: Path, raw_date: str, raw_drive: str) -> Path:
    return kitti_raw_root / raw_date / f"{raw_date}_drive_{raw_drive}_sync"


def _row_transform(row: dict, prefix: str) -> Transform:
    rotation = np.array(row[f"{prefix}_R_cam_from_velo"], dtype=np.float32).reshape(3, 3)
    translation_m = np.array(row[f"{prefix}_t_cam_from_velo_cm"], dtype=np.float32) * CM_TO_M
    return Transform.from_rotation_translation(rotation, translation_m)


class KittiC2LDataset(Dataset):
    """
    Reads one split of the published KITTI-C2L parquet metadata and materializes
    (image, raw point cloud, decalibrated extrinsic) samples via this codebase's
    own :class:`DataPreprocessor`, so preprocessing exactly matches what the
    model was designed for.

    Args:
        parquet_path: path to one of KITTI-C2L's ``{split}.parquet`` files.
        kitti_raw_root: local root containing ``<date>/<date>_drive_<drive>_sync/``.
        preprocessor: configured DataPreprocessor instance (shared with KittiDataset).
        stages: if given, only rows whose ``stage`` is in this set are used
            (default: all 5 difficulty stages, pooled).
        exclude_sequences / include_sequences: optional sequence-id filters,
            used e.g. to carve a validation sequence out of the train split
            (KITTI-C2L publishes train/test only -- see its README's "Open
            items" -- not train/val/test).
    """

    def __init__(
        self,
        parquet_path: str | Path,
        kitti_raw_root: str | Path,
        preprocessor: DataPreprocessor,
        stages: set[int] | None = None,
        exclude_sequences: set[str] | None = None,
        include_sequences: set[str] | None = None,
    ) -> None:
        self.kitti_raw_root = Path(kitti_raw_root)
        self.preprocessor = preprocessor

        table = pq.read_table(parquet_path)
        rows = table.to_pylist()
        if stages is not None:
            rows = [r for r in rows if r["stage"] in stages]
        if exclude_sequences is not None:
            rows = [r for r in rows if r["sequence"] not in exclude_sequences]
        if include_sequences is not None:
            rows = [r for r in rows if r["sequence"] in include_sequences]
        self._rows = rows

    def __len__(self) -> int:
        return len(self._rows)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]

        drive_dir = _drive_dir(self.kitti_raw_root, row["raw_date"], row["raw_drive"])
        camera_id = int(row["camera_id"])
        img_path = drive_dir / f"image_0{camera_id}" / "data" / f"{row['raw_frame_index']:010d}.png"
        lidar_path = drive_dir / "velodyne_points" / "data" / f"{row['raw_frame_index']:010d}.bin"

        img = load_image_rgb(str(img_path))
        raw_pcl = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

        K = np.array(row["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
        T_gt = _row_transform(row, "gt")
        T_init = _row_transform(row, "miscal")  # decalibrated extrinsic
        T_decal = T_init @ T_gt.inverse()  # the correction the model must predict

        img_pp, lidar_map = self.preprocessor(img.copy(), raw_pcl.copy(), T_init.matrix, K.copy())

        return {
            "img": torch.from_numpy(img_pp).permute(2, 0, 1),
            "lidar_map": torch.from_numpy(lidar_map).permute(2, 0, 1),
            "trans": torch.from_numpy(T_decal.translation),
            "rot_mat": torch.from_numpy(T_decal.rotation_matrix),
            "pcl": torch.from_numpy(raw_pcl),
            "metadata": {
                "T_gt": T_gt,
                "T_init": T_init,
                "T_decal": T_decal,
                "K": K,
                "img_name": row["sample_id"],
                "sequence": row["sequence"],
                "frame_index": row["frame_index"],
                "stage": row["stage"],
            },
        }

    @staticmethod
    def collate(samples: list[dict]) -> Batch:
        return KittiDataset.collate(samples)  # identical shape, reuse as-is


class KittiC2LDataModule(L.LightningDataModule):
    """
    Lightning DataModule over the published KITTI-C2L parquet files.

    KITTI-C2L only publishes train/test splits (sequences 00-08 / 09-10 --
    the field-standard protocol). This DataModule carves validation out of
    the train sequences by holding out one whole sequence (never a random
    frame subset: adjacent KITTI frames at 10Hz are near-duplicates, so
    frame-level splitting would leak between train and val).

    Args:
        data_dir:       Root of the KITTI-C2L-Dataset repo checkout (or any
                         directory containing train.parquet / test.parquet).
        kitti_raw_root: Local KITTI raw root, see KittiC2LDataset.
        preprocessor:   Configured DataPreprocessor instance.
        val_sequence:   Sequence id (e.g. "07") held out from train.parquet for validation.
        stages:         Difficulty stages to train/eval on (default: all 5, pooled).
        batch_size, num_workers, pin_memory: as in KittiDataModule.
    """

    def __init__(
        self,
        data_dir: str,
        kitti_raw_root: str,
        preprocessor: Any,
        val_sequence: str = "07",
        stages: list[int] | None = None,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["preprocessor"])
        self._preprocessor = preprocessor
        self._stage_set = set(stages) if stages is not None else None

    def setup(self, stage: str | None = None) -> None:
        data_dir = Path(self.hparams.data_dir)
        val_seq = {self.hparams.val_sequence}

        self.train_ds = KittiC2LDataset(
            data_dir / "train.parquet",
            self.hparams.kitti_raw_root,
            self._preprocessor,
            stages=self._stage_set,
            exclude_sequences=val_seq,
        )
        self.val_ds = KittiC2LDataset(
            data_dir / "train.parquet",
            self.hparams.kitti_raw_root,
            self._preprocessor,
            stages=self._stage_set,
            include_sequences=val_seq,
        )
        self.test_ds = KittiC2LDataset(
            data_dir / "test.parquet",
            self.hparams.kitti_raw_root,
            self._preprocessor,
            stages=self._stage_set,
        )

    def _loader(self, ds: KittiC2LDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size=self.hparams.batch_size,
            num_workers=self.hparams.num_workers,
            pin_memory=self.hparams.pin_memory,
            shuffle=shuffle,
            collate_fn=KittiC2LDataset.collate,
            persistent_workers=self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
