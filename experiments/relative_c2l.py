"""Usual relative-decalibration control on the same frames and GT as KITTI-C2L.

This is an experimental control, not published C2L metadata: training draws
fresh +/-1 degree, +/-10 cm relative errors, as the usual KITTI pipeline does.
Validation/test draws are fixed per published sample seed. Supports both the
unmodified main Batch contract and the branch's T=1 contract explicitly.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import torch

from unical.data.decalibrator import ErrorGenerator
from unical.data.kitti_c2l_dataset import (
    KittiC2LDataModule,
    KittiC2LDataset,
    _drive_dir,
    _row_transform,
)
from unical.utils.geometry import load_image_rgb


class RelativeC2LDataset(KittiC2LDataset):
    """Keep source frames/GT and replace targets with the usual relative protocol."""

    def __init__(
        self,
        *args: Any,
        deterministic: bool,
        relative_seed: int = 0,
        temporal_batch: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.deterministic = deterministic
        self.relative_seed = relative_seed
        self.temporal_batch = temporal_batch
        self.generator = ErrorGenerator(1.0, 10.0)

    def __getitem__(self, idx: int) -> dict:
        row = self._rows[idx]
        gen = None
        if self.deterministic:
            gen = torch.Generator().manual_seed((row["sample_seed"] + self.relative_seed) % (2**63))
        decal = self.generator(generator=gen)
        gt = _row_transform(row, "gt")
        init = decal @ gt
        K = np.array(row["camera_intrinsics"], dtype=np.float32).reshape(3, 3)
        drive = _drive_dir(self.kitti_raw_root, row["raw_date"], row["raw_drive"])
        frame = row["raw_frame_index"]
        image = load_image_rgb(
            str(drive / f"image_0{int(row['camera_id'])}" / "data" / f"{frame:010d}.png")
        )
        points = np.fromfile(
            drive / "velodyne_points" / "data" / f"{frame:010d}.bin", dtype=np.float32
        ).reshape(-1, 4)
        image, depth = self.preprocessor(image.copy(), points.copy(), init.matrix, K.copy())
        image = torch.from_numpy(image).permute(2, 0, 1)
        depth = torch.from_numpy(depth).permute(2, 0, 1)
        points = torch.from_numpy(points)
        meta = {
            "T_gt": gt,
            "T_init": init,
            "T_decal": decal,
            "K": K,
            "img_name": row["sample_id"],
            "sequence": row["sequence"],
            "frame_index": row["frame_index"],
            "stage": row["stage"],
            "protocol": "usual_relative_control",
        }
        return {
            "img": image.unsqueeze(0) if self.temporal_batch else image,
            "lidar_map": depth.unsqueeze(0) if self.temporal_batch else depth,
            "trans": torch.from_numpy(decal.translation),
            "rot_mat": torch.from_numpy(decal.rotation_matrix),
            "pcl": [points] if self.temporal_batch else points,
            "metadata": [meta] if self.temporal_batch else meta,
        }


class RelativeC2LDataModule(KittiC2LDataModule):
    """Use the same sequence split and preprocessor with usual online relative draws."""

    def __init__(self, relative_seed: int = 0, temporal_batch: bool = False, **kwargs: Any) -> None:
        if kwargs.get("stages") is not None:
            kwargs["stages"] = list(kwargs["stages"])
        super().__init__(**kwargs)
        self.relative_seed = relative_seed
        self.temporal_batch = temporal_batch

    def setup(self, stage: str | None = None) -> None:
        directory = Path(self.hparams.data_dir)
        val = {self.hparams.val_sequence}
        common = dict(
            kitti_raw_root=self.hparams.kitti_raw_root,
            preprocessor=self._preprocessor,
            stages=self._stage_set,
            relative_seed=self.relative_seed,
            temporal_batch=self.temporal_batch,
        )
        self.train_ds = RelativeC2LDataset(
            directory / "train.parquet", deterministic=False, exclude_sequences=val, **common
        )
        self.val_ds = RelativeC2LDataset(
            directory / "train.parquet", deterministic=True, include_sequences=val, **common
        )
        self.test_ds = RelativeC2LDataset(directory / "test.parquet", deterministic=True, **common)
