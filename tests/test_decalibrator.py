"""Unit tests for the (dual-pose) decalibration generator."""

import numpy as np
import torch

from unical.data.decalibrator import DualErrorGenerator, ErrorGenerator, compose_net_decalibration
from unical.utils.transform import Transform


def _random_T_gt(seed: int = 0) -> Transform:
    gen = torch.Generator().manual_seed(seed)
    rot = torch.empty(3).uniform_(-0.3, 0.3, generator=gen)
    trans = torch.empty(3).uniform_(-1.0, 1.0, generator=gen)
    from unical.utils.transform import euler_to_transform_matrix

    return Transform(euler_to_transform_matrix(trans, rot).numpy())


def test_zero_camera_range_matches_lidar_only_error_generator():
    """r_range_cam = t_range_cam = 0 must reproduce plain LiDAR-only ErrorGenerator."""
    T_gt = _random_T_gt()
    gen = torch.Generator().manual_seed(42)
    dual = DualErrorGenerator(
        r_range_lidar=1.0, t_range_lidar=10.0, r_range_cam=0.0, t_range_cam=0.0
    )
    T_decal_lidar, T_decal_cam = dual(generator=gen)

    gen2 = torch.Generator().manual_seed(42)
    lidar_only = ErrorGenerator(1.0, 10.0)
    T_decal_lidar_ref = lidar_only(generator=gen2)

    np.testing.assert_allclose(T_decal_lidar.matrix, T_decal_lidar_ref.matrix, atol=1e-6)
    np.testing.assert_allclose(T_decal_cam.matrix, np.eye(4), atol=1e-6)

    T_decal_net = compose_net_decalibration(T_decal_lidar, T_decal_cam, T_gt)
    np.testing.assert_allclose(T_decal_net.matrix, T_decal_lidar.matrix, atol=1e-5)


def test_nonzero_camera_range_produces_nontrivial_composition():
    T_gt = _random_T_gt()
    gen = torch.Generator().manual_seed(7)
    dual = DualErrorGenerator(
        r_range_lidar=1.0, t_range_lidar=10.0, r_range_cam=1.0, t_range_cam=10.0
    )
    T_decal_lidar, T_decal_cam = dual(generator=gen)

    assert not np.allclose(T_decal_cam.matrix, np.eye(4), atol=1e-6)

    T_decal_net = compose_net_decalibration(T_decal_lidar, T_decal_cam, T_gt)

    # T_decal_net @ T_gt must recover T_init = T_decal_lidar @ T_gt @ T_decal_cam.inverse()
    T_init_expected = T_decal_lidar @ T_gt @ T_decal_cam.inverse()
    T_init_actual = T_decal_net @ T_gt
    np.testing.assert_allclose(T_init_actual.matrix, T_init_expected.matrix, atol=1e-5)

    # With a nonzero camera-side error, the net decalibration differs from the
    # LiDAR-only draw (camera error gets folded in via conjugation by T_gt).
    assert not np.allclose(T_decal_net.matrix, T_decal_lidar.matrix, atol=1e-4)
