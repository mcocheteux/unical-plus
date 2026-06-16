"""
Point-cloud ↔ image projection utilities (pure numpy, no device dependency).
"""
from __future__ import annotations

from collections.abc import Callable

import cv2
import numpy as np


def colormap_jet(value: float) -> tuple[int, int, int]:
    value = 2 * value - 1
    blue  = int(min(max(1.5 - abs(2 * value + 1), 0), 1) * 255)
    green = int(min(max(1.5 - abs(2 * value),     0), 1) * 255)
    red   = int(min(max(1.5 - abs(2 * value - 1), 0), 1) * 255)
    return (red, green, blue)


class PointCloudProjector:
    """
    Static helpers for projecting LiDAR point clouds into image space and
    building sparse 2-D lidar maps.
    """

    @staticmethod
    def project(pcl: np.ndarray, K: np.ndarray, T: np.ndarray) -> np.ndarray:
        """
        Project N×4+ point cloud into image coordinates.

        Args:
            pcl: (N, 4+) array [x, y, z, intensity, ...]
            K:   (3, 3) camera intrinsic matrix
            T:   (4, 4) LiDAR-to-camera extrinsic matrix

        Returns:
            (N, 3+) array [u, v, depth, intensity?, ...]
        """
        xyz_h = np.hstack((pcl[:, :3], np.ones((len(pcl), 1), dtype=pcl.dtype))).T  # (4, N)
        cam   = T @ xyz_h               # (4, N)
        img   = K @ cam[:3]             # (3, N)
        img   = img.T                   # (N, 3)
        depth = img[:, 2].copy()
        # Only normalise points in front of the camera (z > 0)
        valid = depth > 0
        img[valid] /= img[valid, 2:3]
        img[~valid] = 0                 # u=v=0, depth=0 → filtered by downstream mask
        uv    = img[:, :2]
        extra = pcl[:, 3:]
        return np.hstack((uv, depth[:, None], extra))

    @staticmethod
    def scale_to_image(pcl: np.ndarray,
                       new_hw: tuple[int, int],
                       old_hw: tuple[int, int]) -> np.ndarray:
        """Scale projected u,v coordinates from old to new image resolution."""
        H_new, W_new = new_hw
        H_old, W_old = old_hw
        out = pcl.copy()
        out[:, 0] *= W_new / W_old
        out[:, 1] *= H_new / H_old
        return out

    @staticmethod
    def to_2d_map(pcl: np.ndarray,
                  hw: tuple[int, int],
                  add_intensity: bool = False) -> np.ndarray:
        """
        Rasterise projected points into a (H, W, C) lidar map.

        Channel 0 → inverse depth (1/z), channel 1 → normalised intensity.
        Points outside the image or with z≤0 are ignored.

        Args:
            pcl:           (N, 3+) projected points [u, v, depth, ...]
            hw:            (H, W)
            add_intensity: include intensity as channel 1

        Returns:
            (H, W, 1) or (H, W, 2) float32 array
        """
        H, W = hw
        C = 2 if (add_intensity and pcl.shape[1] >= 4) else 1
        img = np.zeros((H, W, C), dtype=np.float32)

        mask = (pcl[:, 0] > 0) & (pcl[:, 0] < W) & \
               (pcl[:, 1] > 0) & (pcl[:, 1] < H) & \
               (pcl[:, 2] > 0)
        pcl_valid = pcl[mask]
        if len(pcl_valid) == 0:
            return img
        uv = pcl_valid[:, :2].astype(np.int32)
        img[uv[:, 1], uv[:, 0], 0] = 1.0 / pcl_valid[:, 2]
        if C == 2:
            img[uv[:, 1], uv[:, 0], 1] = pcl_valid[:, 3]
        return img

    @staticmethod
    def draw_on_image(img: np.ndarray,
                      pcl: np.ndarray,
                      radius: int = 1,
                      max_depth: float = 45.0,
                      color_fn: Callable[[float], tuple] = colormap_jet) -> np.ndarray:
        """Overlay projected point cloud on image (uint8 RGB)."""
        if pcl.size == 0:
            return img
        out = img.copy()
        scores = np.clip(pcl[:, 2] / max_depth, 0, 1)
        for i in range(len(pcl)):
            x, y, z = int(pcl[i, 0]), int(pcl[i, 1]), pcl[i, 2]
            if 0 < x < img.shape[1] and 0 < y < img.shape[0] and z > 0:
                cv2.circle(out, (x, y), radius, color_fn(scores[i]), thickness=-1)
        return out


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def min_max_normalize(arr: np.ndarray,
                      axis: tuple[int, ...] | None = None) -> np.ndarray:
    lo = np.min(arr, axis=axis, keepdims=True) if axis else arr.min()
    hi = np.max(arr, axis=axis, keepdims=True) if axis else arr.max()
    return (arr - lo) / (hi - lo + 1e-8)


def imagenet_normalize(img: np.ndarray) -> np.ndarray:
    """Standardise uint8 RGB image to ImageNet statistics."""
    img = img.astype(np.float32) / 255.0
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return (img - mean) / std


def imagenet_denormalize(img: np.ndarray) -> np.ndarray:
    mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
    std  = np.array([0.229, 0.224, 0.225], dtype=np.float32)
    return np.clip((img * std + mean) * 255, 0, 255).astype(np.uint8)


def load_image_rgb(path: str) -> np.ndarray:
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    return img[:, :, ::-1].copy()  # BGR → RGB
