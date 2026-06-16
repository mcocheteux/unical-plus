"""
Data augmentation for paired (RGB image, LiDAR map) inputs.

All operations work on numpy arrays and are device-independent.
"""
from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

# ---------------------------------------------------------------------------
# Photometric distortion
# ---------------------------------------------------------------------------

@dataclass
class PhotometricConfig:
    brightness: float = 0.1255          # delta ratio
    contrast: tuple[float, float] = (0.5, 1.5)
    saturation: tuple[float, float] = (0.80, 1.2)
    hue: float = 18.0


class PhotometricDistortion:
    """Random brightness / contrast / saturation / hue jitter on RGB images."""

    def __init__(self, cfg: PhotometricConfig) -> None:
        self.cfg = cfg

    def __call__(self, img: np.ndarray) -> np.ndarray:
        img = img.astype(np.float32)
        c = self.cfg

        if np.random.randint(2):
            delta = np.random.uniform(-c.brightness * 255, c.brightness * 255)
            img = np.clip(img + delta, 0, 255)

        start_contrast = bool(np.random.randint(2))
        if start_contrast:
            alpha = np.random.uniform(*c.contrast)
            img = np.clip(img * alpha, 0, 255)

        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_RGB2HSV).astype(np.float32)
        if np.random.randint(2):
            img[:, :, 1] = np.clip(img[:, :, 1] * np.random.uniform(*c.saturation), 0, 255)
        if np.random.randint(2):
            img[:, :, 0] = (img[:, :, 0] + np.random.uniform(-c.hue, c.hue)) % 180
        img = cv2.cvtColor(img.astype(np.uint8), cv2.COLOR_HSV2RGB)

        if not start_contrast:
            img = np.clip(img.astype(np.float32) * np.random.uniform(*c.contrast), 0, 255).astype(np.uint8)

        return img.astype(np.uint8)


# ---------------------------------------------------------------------------
# Point-cloud distortion
# ---------------------------------------------------------------------------

@dataclass
class PointCloudConfig:
    dropout_ratio: float = 0.1          # fraction of points that may be removed


class PointCloudDistortion:
    """Random point dropout."""

    def __init__(self, cfg: PointCloudConfig) -> None:
        assert 0.0 <= cfg.dropout_ratio <= 1.0
        self.cfg = cfg

    def __call__(self, pcl: np.ndarray) -> np.ndarray:
        if self.cfg.dropout_ratio > 0 and np.random.randint(2):
            keep = np.random.uniform(1.0 - self.cfg.dropout_ratio, 1.0)
            if keep < 1.0:
                idx = np.random.choice(len(pcl), int(len(pcl) * keep), replace=False)
                return pcl[idx]
        return pcl


# ---------------------------------------------------------------------------
# Spatial distortion (applied jointly to RGB + lidar map)
# ---------------------------------------------------------------------------

@dataclass
class SpatialConfig:
    x_mirror: bool = False
    rotate: float = 0.0                 # degrees; 0 disables
    translate_dw: float = 0.0           # fraction of width
    translate_dh: float = 0.0           # fraction of height
    zoom_out_prob: float = 0.0
    zoom_out_max: float = 2.0
    zoom_out_aspect: tuple[float, float] = (1.0, 1.0)
    zoom_in_prob: float = 0.0
    zoom_in_aspect: tuple[float, float] = (1.0, 1.0)
    zoom_in_crop: tuple[float, float] = (0.5, 1.0)
    crop_box_prob: float = 0.0
    crop_box_width: tuple[float, float] = (0.1, 0.3)
    crop_box_height: tuple[float, float] = (0.1, 0.3)
    crop_box_max_n: int = 3


class SpatialDistortion:
    """Random spatial transforms applied identically to RGB and lidar map."""

    def __init__(self, cfg: SpatialConfig) -> None:
        self.cfg = cfg

    def __call__(self, rgb: np.ndarray, lidar: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        c = self.cfg

        if c.x_mirror and np.random.randint(2):
            rgb   = rgb[:, ::-1].copy()
            lidar = lidar[:, ::-1].copy()

        if c.crop_box_prob > 0 and np.random.rand() < c.crop_box_prob:
            rgb, lidar = self._random_crop_boxes(rgb, lidar)

        if (c.translate_dw > 0 or c.translate_dh > 0) and np.random.randint(2):
            rgb, lidar = self._translate(rgb, lidar)

        if c.rotate > 0 and np.random.randint(2):
            angle = np.random.uniform(-c.rotate, c.rotate)
            center = tuple(np.array(rgb.shape[1::-1]) / 2)
            M = cv2.getRotationMatrix2D(center, angle, 1.0)
            H, W = rgb.shape[:2]
            rgb   = cv2.warpAffine(rgb,   M, (W, H), flags=cv2.INTER_LINEAR)
            lidar = cv2.warpAffine(lidar, M, (W, H), flags=cv2.INTER_NEAREST)
            if rgb.ndim == 2:
                rgb = rgb[:, :, np.newaxis]
            if lidar.ndim == 2:
                lidar = lidar[:, :, np.newaxis]

        prob = np.random.rand()
        if c.zoom_out_prob > 0 and prob < c.zoom_out_prob:
            rgb, lidar = self._zoom_out(rgb, lidar)
        elif c.zoom_in_prob > 0 and prob < c.zoom_out_prob + c.zoom_in_prob:
            rgb, lidar = self._zoom_in(rgb, lidar)

        return rgb, lidar

    # --- helpers -----------------------------------------------------------

    def _random_crop_boxes(self, rgb: np.ndarray, lidar: np.ndarray):
        H, W = rgb.shape[:2]
        n = np.random.randint(1, self.cfg.crop_box_max_n + 1)
        for _ in range(n):
            cx, cy = W * np.random.rand(), H * np.random.rand()
            bw = np.random.uniform(self.cfg.crop_box_width[0] * W, self.cfg.crop_box_width[1] * W)
            bh = np.random.uniform(self.cfg.crop_box_height[0] * H, self.cfg.crop_box_height[1] * H)
            x0, y0 = max(0, int(cx - bw / 2)), max(0, int(cy - bh / 2))
            x1, y1 = min(W, int(cx + bw / 2)), min(H, int(cy + bh / 2))
            rgb[y0:y1, x0:x1] = 127
            lidar[y0:y1, x0:x1] = 0
        return rgb, lidar

    def _translate(self, rgb: np.ndarray, lidar: np.ndarray):
        H, W = rgb.shape[:2]
        dx = np.random.randint(-int(W * self.cfg.translate_dw), int(W * self.cfg.translate_dw) + 1)
        dy = np.random.randint(-int(H * self.cfg.translate_dh), int(H * self.cfg.translate_dh) + 1)
        # dst window (where pixels land after shift)
        dst_x0, dst_y0 = max(0, dx), max(0, dy)
        dst_x1, dst_y1 = min(W, W + dx), min(H, H + dy)
        # src window (corresponding source region before shift)
        src_x0, src_y0 = dst_x0 - dx, dst_y0 - dy
        src_x1, src_y1 = dst_x1 - dx, dst_y1 - dy

        out_rgb = np.full_like(rgb, 127)
        out_rgb[dst_y0:dst_y1, dst_x0:dst_x1] = rgb[src_y0:src_y1, src_x0:src_x1]
        out_lidar = np.zeros_like(lidar)
        out_lidar[dst_y0:dst_y1, dst_x0:dst_x1] = lidar[src_y0:src_y1, src_x0:src_x1]
        return out_rgb, out_lidar

    def _zoom_out(self, rgb: np.ndarray, lidar: np.ndarray):
        H, W = rgb.shape[:2]
        factor = np.random.uniform(1.0, self.cfg.zoom_out_max)
        new_W = int(W / factor)
        aspect = np.random.uniform(self.cfg.zoom_out_aspect[0], self.cfg.zoom_out_aspect[1])
        new_H = min(H, int(new_W / (W / H * aspect)))
        new_W = min(W, new_W)

        small_rgb   = cv2.resize(rgb,   (new_W, new_H))
        small_lidar = cv2.resize(lidar, (new_W, new_H), interpolation=cv2.INTER_NEAREST)
        if small_rgb.ndim == 2:
            small_rgb = small_rgb[:, :, np.newaxis]
        if small_lidar.ndim == 2:
            small_lidar = small_lidar[:, :, np.newaxis]

        canvas_rgb   = np.full((H, W, rgb.shape[2]),   127, dtype=rgb.dtype)
        canvas_lidar = np.zeros((H, W, lidar.shape[2]), dtype=lidar.dtype)
        top, left = (H - new_H) // 2, (W - new_W) // 2
        canvas_rgb[top:top + new_H, left:left + new_W]   = small_rgb
        canvas_lidar[top:top + new_H, left:left + new_W] = small_lidar
        return canvas_rgb, canvas_lidar

    def _zoom_in(self, rgb: np.ndarray, lidar: np.ndarray):
        H, W = rgb.shape[:2]
        cx, cy = W * np.random.rand(), H * np.random.rand()
        aspect = np.random.uniform(self.cfg.zoom_in_aspect[0], self.cfg.zoom_in_aspect[1]) * (W / H)
        lm = np.random.uniform(self.cfg.zoom_in_crop[0], self.cfg.zoom_in_crop[1]) * W / 2
        rm = np.random.uniform(self.cfg.zoom_in_crop[0], self.cfg.zoom_in_crop[1]) * W / 2
        crop_H = (lm + rm) / aspect
        y0 = max(0, int(cy - crop_H * 0.5))
        x0 = max(0, int(cx - lm))
        y1 = min(H, int(cy + crop_H * 0.5))
        x1 = min(W, int(cx + rm))
        if x1 <= x0 or y1 <= y0:
            return rgb, lidar
        crop_rgb   = rgb[y0:y1, x0:x1]
        crop_lidar = lidar[y0:y1, x0:x1]
        out_rgb   = cv2.resize(crop_rgb,   (W, H))
        out_lidar = cv2.resize(crop_lidar, (W, H), interpolation=cv2.INTER_NEAREST)
        if out_rgb.ndim == 2:
            out_rgb = out_rgb[:, :, np.newaxis]
        if out_lidar.ndim == 2:
            out_lidar = out_lidar[:, :, np.newaxis]
        return out_rgb, out_lidar


# ---------------------------------------------------------------------------
# Composite augmentor
# ---------------------------------------------------------------------------

@dataclass
class AugmentationConfig:
    photometric: PhotometricConfig | None = None
    point_cloud: PointCloudConfig | None  = None
    spatial:     SpatialConfig | None     = None


class Augmentor:
    """Combines photometric, point-cloud, and spatial distortions."""

    def __init__(self, cfg: AugmentationConfig = AugmentationConfig()) -> None:
        self.photo = PhotometricDistortion(cfg.photometric) if cfg.photometric else None
        self.pcl   = PointCloudDistortion(cfg.point_cloud) if cfg.point_cloud else None
        self.spatial = SpatialDistortion(cfg.spatial)       if cfg.spatial    else None
