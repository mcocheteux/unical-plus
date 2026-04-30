"""
KITTI raw dataset for camera-LiDAR calibration.

Each sample is a (image, lidar_map) pair with a random decalibration applied
to the LiDAR projection.  The model must predict that decalibration error.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, NamedTuple, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Dataset

from unical.data.decalibrator import ErrorGenerator
from unical.data.preprocessor import DataPreprocessor
from unical.utils.geometry import load_image_rgb
from unical.utils.transform import Transform


# ---------------------------------------------------------------------------
# Batch type
# ---------------------------------------------------------------------------

class Batch(NamedTuple):
    """
    Inputs fed to UniCal.

    img:        (B, C_img, H, W)   — normalised camera image
    lidar_map:  (B, C_lid, H, W)   — normalised sparse depth map
    target_reg: tuple[(B, 3), (B, 3)] — (translation, euler) decal target
    pcl:        (B, N, 4)          — raw padded LiDAR scan (for spatial loss)
    metadata:   list of per-sample dicts
    """
    img:        torch.Tensor
    lidar_map:  torch.Tensor
    target_reg: Tuple[torch.Tensor, torch.Tensor]
    pcl:        torch.Tensor
    metadata:   List[Dict[str, Any]]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

Split = List[Tuple[str, List[int]]]   # [("2011_09_26", [1, 2, ...]), ...]


class KittiDataset(Dataset):
    """
    KITTI raw dataset loader.

    Args:
        data_dir:     Root of the KITTI raw dataset
                      (contains date folders like 2011_09_26/).
        split:        List of (date_str, [drive_ids]) tuples.
        preprocessor: Preprocessing pipeline.
        decalibrator: Random decalibration generator.
    """

    def __init__(
        self,
        data_dir: str | Path,
        split: Split,
        preprocessor: DataPreprocessor,
        decalibrator: ErrorGenerator,
    ) -> None:
        self.data_dir     = Path(data_dir)
        self.preprocessor = preprocessor
        self.decalibrator = decalibrator

        self._samples: List[Tuple[str, int, int]] = []  # (date, drive, frame_id)
        self._date_meta: Dict[str, Dict] = {}

        self._parse(split)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _parse(self, split: Split) -> None:
        for date, drives in split:
            date_dir = self.data_dir / date
            self._date_meta[date] = self._read_calibration(date_dir)
            for drive in drives:
                drive_dir = date_dir / f"{date}_drive_{drive:04d}_sync"
                img_dir   = drive_dir / "image_02" / "data"
                lidar_dir = drive_dir / "velodyne_points" / "data"
                for img_path in sorted(p for p in img_dir.glob("*.png") if not p.name.startswith(".")):
                    fid = int(img_path.stem)
                    lid_path = lidar_dir / f"{fid:010d}.bin"
                    if lid_path.exists():
                        self._samples.append((date, drive, fid))

    @staticmethod
    def _read_calibration(date_dir: Path) -> Dict:
        """Parse KITTI raw calibration files for a date folder."""
        # camera intrinsics
        K = P = None
        with open(date_dir / "calib_cam_to_cam.txt", encoding="utf-8") as f:
            for line in f:
                if line.startswith("P_rect_02:"):
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
        T_gt = Transform.from_rotation_translation(R, t)
        return {"K": K, "P": P, "T_gt": T_gt}

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------

    def __len__(self) -> int:
        return len(self._samples)

    def __getitem__(self, idx: int) -> Dict:
        date, drive, fid = self._samples[idx]
        meta = self._date_meta[date]
        K    = meta["K"]
        T_gt = meta["T_gt"]   # ground-truth LiDAR→cam

        img_path = (
            self.data_dir / date / f"{date}_drive_{drive:04d}_sync"
            / "image_02" / "data" / f"{fid:010d}.png"
        )
        lidar_path = (
            self.data_dir / date / f"{date}_drive_{drive:04d}_sync"
            / "velodyne_points" / "data" / f"{fid:010d}.bin"
        )

        img      = load_image_rgb(str(img_path))
        raw_pcl  = np.fromfile(str(lidar_path), dtype=np.float32).reshape(-1, 4)

        # Sample a random decalibration error
        T_decal   = self.decalibrator()        # Transform
        T_init    = T_decal @ T_gt             # decalibrated extrinsic

        # Preprocess image + project lidar with decalibrated extrinsic
        img_pp, lidar_map = self.preprocessor(
            img.copy(), raw_pcl.copy(), T_init.matrix, K.copy()
        )

        # Regression target: the decalibration we want to predict
        t_target, r_target = T_decal.to_euler_components()  # numpy (3,)

        return {
            "img":       torch.from_numpy(img_pp).permute(2, 0, 1),    # (C, H, W)
            "lidar_map": torch.from_numpy(lidar_map).permute(2, 0, 1), # (C, H, W)
            "trans":     torch.from_numpy(t_target),                    # (3,)
            "rot":       torch.from_numpy(r_target),                    # (3,)
            "pcl":       torch.from_numpy(raw_pcl),                     # (N, 4)
            "metadata": {
                "T_gt":    T_gt,      # Transform (numpy) — ground truth
                "T_init":  T_init,    # Transform (numpy) — decalibrated
                "T_decal": T_decal,   # Transform (numpy) — target
                "K":       K,
                "img_name": f"{fid:010d}",
            },
        }

    # ------------------------------------------------------------------
    # Collate
    # ------------------------------------------------------------------

    @staticmethod
    def collate(samples: List[Dict]) -> Batch:
        img       = torch.stack([s["img"]       for s in samples])
        lidar_map = torch.stack([s["lidar_map"] for s in samples])
        trans     = torch.stack([s["trans"]     for s in samples])
        rot       = torch.stack([s["rot"]       for s in samples])
        # Pad point clouds to the same length (pad value = 0)
        pcl = torch.nn.utils.rnn.pad_sequence(
            [s["pcl"] for s in samples], batch_first=True
        )
        metadata = [s["metadata"] for s in samples]
        return Batch(
            img=img,
            lidar_map=lidar_map,
            target_reg=(trans, rot),
            pcl=pcl,
            metadata=metadata,
        )
