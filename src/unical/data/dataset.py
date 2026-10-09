"""
KITTI raw dataset for camera-LiDAR calibration.

Each sample is a (image, lidar_map) pair with a random decalibration applied
to the LiDAR projection.  The model must predict that decalibration error.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import torch
from torch.utils.data import Dataset

from unical.data.decalibrator import (
    DualErrorGenerator,
    ErrorGenerator,
    compose_net_decalibration,
)
from unical.data.preprocessor import DataPreprocessor
from unical.utils.geometry import load_image_rgb
from unical.utils.transform import Transform

# ---------------------------------------------------------------------------
# Batch type
# ---------------------------------------------------------------------------


class Batch(NamedTuple):
    """
    Inputs fed to UniCal.

    img:        (B, T, C_img, H, W) — normalised camera image, T = window length
    lidar_map:  (B, T, C_lid, H, W) — normalised sparse depth map
    target_reg: tuple[(B, 3), (B, 3, 3)] — (translation, rotation_matrix) decal
                target; one prediction per *window*, not per frame, since the
                decalibration is shared across all T frames.
    pcl:        list of T tensors, each (B, N_max, 4) — raw padded LiDAR scan
                at that window position (for the spatial loss).
    metadata:   list of T per-position lists of per-sample dicts (i.e.
                metadata[t][i] is sample i's metadata for frame t).
    """

    img: torch.Tensor
    lidar_map: torch.Tensor
    target_reg: tuple[torch.Tensor, torch.Tensor]
    pcl: list[torch.Tensor]
    metadata: list[list[dict[str, Any]]]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

Split = list[tuple[str, list[int]]]  # [("2011_09_26", [1, 2, ...]), ...]


class KittiDataset(Dataset):
    """
    KITTI raw dataset loader.

    Args:
        data_dir:        Root of the KITTI raw dataset
                          (contains date folders like 2011_09_26/).
        split:            List of (date_str, [drive_ids]) tuples.
        preprocessor:     Preprocessing pipeline.
        decalibrator:     Random decalibration generator (camera + LiDAR sides).
        sequence_length:  Number of consecutive frames per sample. ``1``
                          (default) reproduces the original single-frame
                          behaviour exactly — same sample count and order.
        frame_stride:     Gap between sampled frames within a window.
        deterministic:    Seed decalibration draws per sample index (val/test).
        seed:             Base seed for deterministic mode.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Split,
        preprocessor: DataPreprocessor,
        decalibrator: DualErrorGenerator | ErrorGenerator,
        sequence_length: int = 1,
        frame_stride: int = 1,
        deterministic: bool = False,
        seed: int = 0,
    ) -> None:
        if sequence_length < 1:
            raise ValueError(f"sequence_length must be >= 1, got {sequence_length}")
        if frame_stride < 1:
            raise ValueError(f"frame_stride must be >= 1, got {frame_stride}")

        self.data_dir = Path(data_dir)
        self.preprocessor = preprocessor
        self.decalibrator = decalibrator
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self.deterministic = deterministic
        self.seed = seed

        # (date, drive, [fid_0, ..., fid_{T-1}]) — a window of T contiguous frames.
        self._samples: list[tuple[str, int, list[int]]] = []
        self._date_meta: dict[str, dict] = {}

        self._parse(split)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse(self, split: Split) -> None:
        T, stride = self.sequence_length, self.frame_stride
        for date, drives in split:
            date_dir = self.data_dir / date
            self._date_meta[date] = self._read_calibration(date_dir)
            for drive in drives:
                drive_dir = date_dir / f"{date}_drive_{drive:04d}_sync"
                img_dir = drive_dir / "image_02" / "data"
                lidar_dir = drive_dir / "velodyne_points" / "data"
                valid_fids = []
                for img_path in sorted(
                    p for p in img_dir.glob("*.png") if not p.name.startswith(".")
                ):
                    fid = int(img_path.stem)
                    lid_path = lidar_dir / f"{fid:010d}.bin"
                    if lid_path.exists():
                        valid_fids.append(fid)

                # Only sampled frames must exist; missing intermediate frames
                # are harmless when stride > 1.
                valid_set = set(valid_fids)
                for fid in valid_fids:
                    window = [fid + k * stride for k in range(T)]
                    if all(frame in valid_set for frame in window):
                        self._samples.append((date, drive, window))

    @staticmethod
    def _read_calibration(date_dir: Path) -> dict:
        """Parse KITTI raw calibration files for a date folder."""
        # camera intrinsics
        K = P = None
        R_rect = np.eye(3, dtype=np.float32)
        with open(date_dir / "calib_cam_to_cam.txt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("R_rect_00:"):
                    R_rect = np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
                elif line.startswith("P_rect_02:"):
                    vals = np.array(line.split()[1:], dtype=np.float32).reshape(3, 4)
                    P = vals
                    K = vals[:3, :3]

        # LiDAR→camera extrinsic
        R = t = None
        with open(date_dir / "calib_velo_to_cam.txt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("R:"):
                    R = np.array(line.split()[1:], dtype=np.float32).reshape(3, 3)
                elif line.startswith("T:"):
                    t = np.array(line.split()[1:], dtype=np.float32)
        if K is None or R is None or t is None:
            raise ValueError(f"Incomplete KITTI calibration in {date_dir}")
        # image_02 is rectified and offset from camera 0 by the stereo baseline.
        # Move P's translation into the extrinsic so projection with K matches P.
        T_gt = Transform.from_rotation_translation(
            R_rect @ R, R_rect @ t + np.linalg.solve(K, P[:, 3])
        )
        return {"K": K, "P": P, "T_gt": T_gt}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> dict:
        date, drive, fids = self._samples[idx]
        meta = self._date_meta[date]
        K = meta["K"]
        T_gt = meta["T_gt"]  # ground-truth LiDAR→cam

        drive_dir = self.data_dir / date / f"{date}_drive_{drive:04d}_sync"

        # One decalibration draw per window, shared by every frame in it —
        # the physical miscalibration is constant across a short time span.
        # NOTE: spatial augmentation (crop/flip/zoom in `Augmentor.spatial`)
        # uses the global numpy RNG rather than an injectable generator, so
        # it is not guaranteed to be identical across a window's frames if
        # enabled. Photometric jitter being independent per frame is fine
        # (realistic per-frame exposure/lighting noise); spatial augmentation
        # is disabled by default in this repo's configs, so this only matters
        # if a caller explicitly enables it.
        gen = (
            torch.Generator().manual_seed(self.seed * 1_000_003 + idx)
            if self.deterministic
            else None
        )
        draw = self.decalibrator(generator=gen)
        T_decal = compose_net_decalibration(*draw, T_gt) if isinstance(draw, tuple) else draw
        T_init = T_decal @ T_gt  # decalibrated extrinsic, shared across the window

        imgs, lidar_maps, pcls, metadatas = [], [], [], []
        for fid in fids:
            img_path = drive_dir / "image_02" / "data" / f"{fid:010d}.png"
            lidar_path = drive_dir / "velodyne_points" / "data" / f"{fid:010d}.bin"

            img = load_image_rgb(str(img_path))
            raw_pcl = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

            img_pp, lidar_map = self.preprocessor(
                img.copy(), raw_pcl.copy(), T_init.matrix, K.copy()
            )

            imgs.append(torch.from_numpy(img_pp).permute(2, 0, 1))  # (C, H, W)
            lidar_maps.append(torch.from_numpy(lidar_map).permute(2, 0, 1))  # (C, H, W)
            pcls.append(torch.from_numpy(raw_pcl))  # (N, 4)
            metadatas.append(
                {
                    "T_gt": T_gt,  # Transform — ground truth (same for whole window)
                    "T_init": T_init,  # Transform — decalibrated (same for whole window)
                    "T_decal": T_decal,  # Transform — target (same for whole window)
                    "K": K,
                    "img_name": f"{fid:010d}",
                }
            )

        # Regression target: the decalibration we want to predict, as a translation
        # vector + rotation matrix (matrix target avoids Euler-convention ambiguity).
        # One target per window, not per frame.
        t_target = T_decal.translation  # numpy (3,)
        R_target = T_decal.rotation_matrix  # numpy (3, 3)

        return {
            "img": torch.stack(imgs),  # (T, C, H, W)
            "lidar_map": torch.stack(lidar_maps),  # (T, C, H, W)
            "trans": torch.from_numpy(t_target),  # (3,)
            "rot_mat": torch.from_numpy(R_target),  # (3, 3)
            "pcl": pcls,  # list of T tensors, each (N, 4)
            "metadata": metadatas,  # list of T dicts
        }

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate(samples: list[dict]) -> Batch:
        img = torch.stack([s["img"] for s in samples])  # (B, T, C, H, W)
        lidar_map = torch.stack([s["lidar_map"] for s in samples])  # (B, T, C, H, W)
        trans = torch.stack([s["trans"] for s in samples])
        rot_mat = torch.stack([s["rot_mat"] for s in samples])
        T = len(samples[0]["pcl"])
        # Pad point clouds to the same length within each window position.
        pcl = [
            torch.nn.utils.rnn.pad_sequence([s["pcl"][t] for s in samples], batch_first=True)
            for t in range(T)
        ]
        metadata = [[s["metadata"][t] for s in samples] for t in range(T)]
        return Batch(
            img=img,
            lidar_map=lidar_map,
            target_reg=(trans, rot_mat),
            pcl=pcl,
            metadata=metadata,
        )
