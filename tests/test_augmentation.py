"""Unit tests for data augmentation utilities."""

from __future__ import annotations

import numpy as np
import pytest

from unical.utils.augmentation import (
    AugmentationConfig,
    Augmentor,
    PhotometricConfig,
    PhotometricDistortion,
    PointCloudConfig,
    PointCloudDistortion,
    SpatialConfig,
    SpatialDistortion,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


def _rgb(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, (h, w, 3), dtype=np.uint8)


def _lidar(h: int = 64, w: int = 64) -> np.ndarray:
    return np.random.default_rng(1).random((h, w, 1)).astype(np.float32)


def _pcl(n: int = 100) -> np.ndarray:
    return np.random.default_rng(2).random((n, 4)).astype(np.float32)


# ---------------------------------------------------------------------------
# PhotometricDistortion
# ---------------------------------------------------------------------------


def test_photometric_output_shape_and_dtype() -> None:
    dist = PhotometricDistortion(PhotometricConfig())
    out = dist(_rgb())
    assert out.shape == (64, 64, 3)
    assert out.dtype == np.uint8


def test_photometric_values_in_valid_range() -> None:
    dist = PhotometricDistortion(PhotometricConfig())
    np.random.seed(7)
    for _ in range(10):
        out = dist(_rgb())
        assert int(out.min()) >= 0 and int(out.max()) <= 255


def test_photometric_identity_config_no_crash() -> None:
    cfg = PhotometricConfig(brightness=0.0, contrast=(1.0, 1.0), saturation=(1.0, 1.0), hue=0.0)
    dist = PhotometricDistortion(cfg)
    out = dist(_rgb())
    assert out.shape == (64, 64, 3)


def test_photometric_hue_stays_in_range() -> None:
    """Hue channel (OpenCV HSV) must remain in [0, 180]."""
    import cv2

    dist = PhotometricDistortion(PhotometricConfig(hue=18.0))
    np.random.seed(3)
    for _ in range(10):
        out = dist(_rgb())
        hsv = cv2.cvtColor(out, cv2.COLOR_RGB2HSV)
        assert int(hsv[:, :, 0].max()) <= 180


# ---------------------------------------------------------------------------
# PointCloudDistortion
# ---------------------------------------------------------------------------


def test_pointcloud_zero_dropout_returns_all() -> None:
    dist = PointCloudDistortion(PointCloudConfig(dropout_ratio=0.0))
    pcl = _pcl(100)
    assert len(dist(pcl)) == 100


def test_pointcloud_dropout_can_reduce_count() -> None:
    dist = PointCloudDistortion(PointCloudConfig(dropout_ratio=0.5))
    np.random.seed(42)
    sizes = {len(dist(_pcl(200))) for _ in range(20)}
    assert any(s < 200 for s in sizes), "dropout_ratio=0.5 should occasionally reduce point count"


def test_pointcloud_dropout_never_exceeds_original() -> None:
    dist = PointCloudDistortion(PointCloudConfig(dropout_ratio=0.5))
    pcl = _pcl(100)
    np.random.seed(0)
    for _ in range(10):
        assert len(dist(pcl.copy())) <= 100


def test_pointcloud_invalid_ratio_raises() -> None:
    with pytest.raises(AssertionError):
        PointCloudDistortion(PointCloudConfig(dropout_ratio=1.5))


# ---------------------------------------------------------------------------
# SpatialDistortion
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "cfg",
    [
        SpatialConfig(x_mirror=True),
        SpatialConfig(translate_dw=0.15, translate_dh=0.15),
        SpatialConfig(crop_box_prob=1.0, crop_box_max_n=2),
        SpatialConfig(zoom_out_prob=1.0, zoom_out_max=2.0),
        SpatialConfig(zoom_in_prob=1.0, zoom_in_crop=(0.5, 0.8)),
        SpatialConfig(rotate=15.0),
    ],
)
def test_spatial_preserves_shape(cfg: SpatialConfig) -> None:
    np.random.seed(0)
    dist = SpatialDistortion(cfg)
    rgb, lid = _rgb(), _lidar()
    for _ in range(3):
        out_rgb, out_lid = dist(rgb.copy(), lid.copy())
        assert out_rgb.shape == rgb.shape, f"RGB shape mismatch for {cfg}"
        assert out_lid.shape == lid.shape, f"LiDAR shape mismatch for {cfg}"


def test_translate_actually_moves_content() -> None:
    """A non-zero translation must change pixel values (catches the no-op bug)."""
    np.random.seed(0)
    cfg = SpatialConfig(translate_dw=0.3, translate_dh=0.3)
    dist = SpatialDistortion(cfg)
    rgb = np.full((64, 64, 3), 200, dtype=np.uint8)
    rgb[20:40, 20:40] = 50  # distinct block in the centre
    changed = False
    for _ in range(20):
        out_rgb, _ = dist(rgb.copy(), _lidar())
        if not np.array_equal(out_rgb, rgb):
            changed = True
            break
    assert changed, "_translate must shift pixel content, not copy in-place"


def test_mirror_is_horizontal_flip() -> None:
    np.random.seed(1)
    dist = SpatialDistortion(SpatialConfig(x_mirror=True))
    rgb = _rgb()
    # Run until mirror branch is taken (50 % chance each call)
    for _ in range(50):
        out, _ = dist(rgb.copy(), _lidar())
        if np.array_equal(out, rgb[:, ::-1]):
            return  # confirmed
    pytest.fail("x_mirror never produced a horizontal flip in 50 attempts")


# ---------------------------------------------------------------------------
# Augmentor (composite)
# ---------------------------------------------------------------------------


def test_augmentor_default_config_all_none() -> None:
    aug = Augmentor(AugmentationConfig())
    assert aug.photo is None
    assert aug.pcl is None
    assert aug.spatial is None


def test_augmentor_components_instantiated_when_configured() -> None:
    cfg = AugmentationConfig(
        photometric=PhotometricConfig(),
        point_cloud=PointCloudConfig(dropout_ratio=0.1),
        spatial=SpatialConfig(x_mirror=True),
    )
    aug = Augmentor(cfg)
    assert aug.photo is not None
    assert aug.pcl is not None
    assert aug.spatial is not None
