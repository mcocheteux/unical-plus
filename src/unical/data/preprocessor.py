"""
Preprocessor: takes a raw RGB image + raw LiDAR scan and produces the
normalised tensors that the model expects as inputs.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from unical.utils.augmentation import AugmentationConfig, Augmentor
from unical.utils.geometry import (
    PointCloudProjector,
    imagenet_normalize,
    min_max_normalize,
)


@dataclass
class PreprocessorConfig:
    width:            int  = 512
    height:           int  = 512
    grayscale:        bool = False
    add_intensity:    bool = False
    min_depth:        float = 2.0       # metres — filter near points
    max_depth:        float = 80.0      # metres — filter far  points
    augmentation:     AugmentationConfig = field(default_factory=AugmentationConfig)


class DataPreprocessor:
    """
    Prepares a (camera, lidar) pair for the model.

    Returns:
        img:       (H, W, 3|1) float32, normalised
        lidar_map: (H, W, 1|2) float32, normalised
    """

    def __init__(self, cfg: PreprocessorConfig) -> None:
        self.cfg = cfg
        self.aug = Augmentor(cfg.augmentation)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def __call__(
        self,
        img:   np.ndarray,   # (H_orig, W_orig, 3) uint8 RGB
        pcl:   np.ndarray,   # (N, 4) float32 [x, y, z, intensity]
        T:     np.ndarray,   # (4, 4) LiDAR→camera extrinsic (decalibrated)
        K:     np.ndarray,   # (3, 3) intrinsic
        D:     np.ndarray | None = None,  # distortion coefficients
    ) -> tuple[np.ndarray, np.ndarray]:
        # 1. Optional undistort
        if D is not None:
            img = self._undistort(img, K, D)

        orig_hw = img.shape[:2]

        # 2. Resize image
        img = cv2.resize(img, (self.cfg.width, self.cfg.height))
        new_hw = img.shape[:2]

        # 3. Photometric augmentation on image
        if self.aug.photo is not None:
            img = self.aug.photo(img)

        # 4. Grayscale conversion
        if self.cfg.grayscale:
            img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)[:, :, np.newaxis]

        # 5. Project LiDAR → image space
        pcl_proj = PointCloudProjector.project(pcl, K, T)
        if orig_hw != new_hw:
            pcl_proj = PointCloudProjector.scale_to_image(pcl_proj, new_hw, orig_hw)

        # 6. Distance filter
        depth_mask = (pcl_proj[:, 2] > self.cfg.min_depth) & \
                     (pcl_proj[:, 2] < self.cfg.max_depth)
        pcl_proj = pcl_proj[depth_mask]

        # 7. Point-cloud augmentation
        if self.aug.pcl is not None:
            # mask to valid image region, augment, then keep all
            valid = (
                (pcl_proj[:, 0] > 0) & (pcl_proj[:, 0] < self.cfg.width)  &
                (pcl_proj[:, 1] > 0) & (pcl_proj[:, 1] < self.cfg.height) &
                (pcl_proj[:, 2] > 0)
            )
            pcl_proj[valid] = self.aug.pcl(pcl_proj[valid])

        # 8. Rasterise to 2-D map
        lidar_map = PointCloudProjector.to_2d_map(
            pcl_proj, new_hw, add_intensity=self.cfg.add_intensity
        )

        # 9. Spatial augmentation
        if self.aug.spatial is not None:
            img, lidar_map = self.aug.spatial(img, lidar_map)

        # 10. Normalise
        img       = (imagenet_normalize(img) if not self.cfg.grayscale
                     else min_max_normalize(img, axis=(0, 1))).astype(np.float32)
        lidar_map = min_max_normalize(lidar_map, axis=(0, 1)).astype(np.float32)

        return img, lidar_map

    @staticmethod
    def _undistort(img: np.ndarray, K: np.ndarray, D: np.ndarray) -> np.ndarray:
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, D, np.eye(3), K, (img.shape[1], img.shape[0]), cv2.CV_32FC1
        )
        return cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
