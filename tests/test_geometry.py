"""Unit tests for point-cloud projection and normalisation utilities."""

from __future__ import annotations

import numpy as np

from unical.utils.geometry import (
    PointCloudProjector,
    imagenet_denormalize,
    imagenet_normalize,
    min_max_normalize,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _K() -> np.ndarray:
    """Pinhole camera: focal=100, principal=(50, 50)."""
    return np.array([[100.0, 0.0, 50.0], [0.0, 100.0, 50.0], [0.0, 0.0, 1.0]], dtype=np.float64)


def _I4() -> np.ndarray:
    return np.eye(4, dtype=np.float64)


# ---------------------------------------------------------------------------
# PointCloudProjector.project
# ---------------------------------------------------------------------------


def test_project_output_shape() -> None:
    pcl = np.random.default_rng(0).random((50, 4)).astype(np.float32)
    pcl[:, 2] = np.abs(pcl[:, 2]) + 1.0  # all points in front of camera
    out = PointCloudProjector.project(pcl, _K(), _I4())
    assert out.shape == (50, 4)


def test_project_principal_point() -> None:
    """A point on the optical axis at z=10 should project to the principal point."""
    pcl = np.array([[0.0, 0.0, 10.0, 1.0]])
    out = PointCloudProjector.project(pcl, _K(), _I4())
    assert abs(out[0, 0] - 50.0) < 1e-4, "u should equal cx"
    assert abs(out[0, 1] - 50.0) < 1e-4, "v should equal cy"
    assert abs(out[0, 2] - 10.0) < 1e-4, "depth should be preserved"


def test_project_behind_camera_zeroed() -> None:
    """Points with z ≤ 0 must have depth ≤ 0 so downstream masks filter them out."""
    pcl = np.array([[0.0, 0.0, -5.0, 1.0]])
    out = PointCloudProjector.project(pcl, _K(), _I4())
    assert out[0, 2] <= 0.0


def test_project_extra_columns_preserved() -> None:
    """Columns beyond xyz (intensity, ring, etc.) must pass through unchanged."""
    pcl = np.array([[0.0, 0.0, 5.0, 0.7, 42.0]])
    out = PointCloudProjector.project(pcl, _K(), _I4())
    assert out.shape == (1, 5)
    assert abs(out[0, 3] - 0.7) < 1e-6
    assert abs(out[0, 4] - 42.0) < 1e-6


def test_project_depth_scales_with_distance() -> None:
    """Closer points should have smaller depth values."""
    pcl = np.array([[0.0, 0.0, 2.0, 1.0], [0.0, 0.0, 5.0, 1.0], [0.0, 0.0, 10.0, 1.0]])
    out = PointCloudProjector.project(pcl, _K(), _I4())
    assert out[0, 2] < out[1, 2] < out[2, 2]


# ---------------------------------------------------------------------------
# PointCloudProjector.to_2d_map
# ---------------------------------------------------------------------------


def test_to_2d_map_empty_cloud_returns_zeros() -> None:
    pcl = np.zeros((0, 3), dtype=np.float32)
    out = PointCloudProjector.to_2d_map(pcl, (64, 64))
    assert out.shape == (64, 64, 1)
    assert out.sum() == 0.0


def test_to_2d_map_single_point_inverse_depth() -> None:
    """A point at depth=5 should write 1/5 = 0.2 into the correct pixel."""
    pcl = np.array([[10.0, 10.0, 5.0]])  # u=10, v=10, depth=5
    out = PointCloudProjector.to_2d_map(pcl, (64, 64))
    assert out.shape == (64, 64, 1)
    assert abs(out[10, 10, 0] - 0.2) < 1e-5


def test_to_2d_map_intensity_channel() -> None:
    """With add_intensity=True a 4-col cloud must produce a 2-channel output."""
    pcl = np.array([[10.0, 10.0, 5.0, 0.8]])
    out = PointCloudProjector.to_2d_map(pcl, (64, 64), add_intensity=True)
    assert out.shape == (64, 64, 2)
    assert abs(out[10, 10, 1] - 0.8) < 1e-5


def test_to_2d_map_out_of_bounds_ignored() -> None:
    pcl = np.array(
        [
            [-1.0, 10.0, 5.0],  # u < 0
            [10.0, -1.0, 5.0],  # v < 0
            [65.0, 10.0, 5.0],
        ]
    )  # u >= W
    out = PointCloudProjector.to_2d_map(pcl, (64, 64))
    assert out.sum() == 0.0


def test_to_2d_map_inverse_depth_monotone() -> None:
    """Closer points (smaller depth) must produce larger inverse-depth values."""
    near = np.array([[32.0, 32.0, 2.0]])
    far = np.array([[32.0, 32.0, 10.0]])
    out_near = PointCloudProjector.to_2d_map(near, (64, 64))
    out_far = PointCloudProjector.to_2d_map(far, (64, 64))
    assert out_near[32, 32, 0] > out_far[32, 32, 0]


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------


def test_min_max_normalize_range() -> None:
    arr = np.array([2.0, 4.0, 6.0, 8.0])
    out = min_max_normalize(arr)
    assert abs(float(out.min())) < 1e-6
    assert abs(float(out.max()) - 1.0) < 1e-6


def test_min_max_normalize_constant_no_nan() -> None:
    out = min_max_normalize(np.ones((4, 4)))
    assert np.isfinite(out).all()


def test_imagenet_normalize_shape_and_dtype() -> None:
    img = np.full((32, 32, 3), 128, dtype=np.uint8)
    out = imagenet_normalize(img)
    assert out.shape == (32, 32, 3)
    assert out.dtype == np.float32


def test_imagenet_roundtrip() -> None:
    """imagenet_denormalize(imagenet_normalize(img)) should recover the original
    pixel values up to uint8 quantisation (±1 LSB)."""
    rng = np.random.default_rng(0)
    img = rng.integers(0, 256, (32, 32, 3), dtype=np.uint8)
    recovered = imagenet_denormalize(imagenet_normalize(img))
    diff = np.abs(recovered.astype(np.int32) - img.astype(np.int32))
    assert int(diff.max()) <= 1
