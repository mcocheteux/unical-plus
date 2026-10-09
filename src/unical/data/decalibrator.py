"""
Random decalibration generator.

Produces a random rigid-body error T_decal sampled uniformly within
configurable rotation (degrees) and translation (centimetres) ranges.
"""

from __future__ import annotations

import math

import torch

from unical.utils.transform import Transform, euler_to_transform_matrix


class ErrorGenerator:
    """
    Sample a random decalibration transform T_decal ~ Uniform(−range, +range).

    Args:
        r_range: Max absolute rotation error in **degrees** (applied per axis).
        t_range: Max absolute translation error in **centimetres** (applied per axis).
    """

    def __init__(self, r_range: float, t_range: float) -> None:
        self._r_rad = math.radians(r_range)
        self._t_m = t_range / 100.0  # cm → m

    def __call__(self, generator: torch.Generator | None = None) -> Transform:
        """Sample a decalibration. Pass a seeded ``generator`` for reproducibility
        (used by the val/test splits so their decalibrations are deterministic)."""
        rot = torch.empty(3).uniform_(-self._r_rad, self._r_rad, generator=generator)
        trans = torch.empty(3).uniform_(-self._t_m, self._t_m, generator=generator)
        T = euler_to_transform_matrix(trans, rot).cpu().numpy()
        return Transform(T)


class DualErrorGenerator:
    """
    Sample independent camera-side and LiDAR-side decalibrations.

    Each draw is expressed in its sensor's local coordinate frame. If the
    sensor-to-rig mounting pose is right-perturbed by its draw, the resulting
    LiDAR-to-camera transform is ``D_cam.inverse() @ T_gt @ D_lidar``.
    The dataset composes the draws into one relative target using ``T_gt``.
    Use ``ErrorGenerator`` for the legacy camera-frame relative perturbation.

    Args:
        r_range_lidar: Max absolute LiDAR-side rotation error in degrees.
        t_range_lidar: Max absolute LiDAR-side translation error in centimetres.
        r_range_cam:   Max absolute camera-side rotation error in degrees.
        t_range_cam:   Max absolute camera-side translation error in centimetres.
    """

    def __init__(
        self,
        r_range_lidar: float,
        t_range_lidar: float,
        r_range_cam: float = 0.0,
        t_range_cam: float = 0.0,
    ) -> None:
        self._lidar_gen = ErrorGenerator(r_range_lidar, t_range_lidar)
        self._cam_gen = ErrorGenerator(r_range_cam, t_range_cam)

    def __call__(self, generator: torch.Generator | None = None) -> tuple[Transform, Transform]:
        """Sample (T_decal_lidar, T_decal_cam). Pass a seeded ``generator``
        for reproducibility (used by the val/test splits)."""
        T_decal_lidar = self._lidar_gen(generator=generator)
        T_decal_cam = self._cam_gen(generator=generator)
        return T_decal_lidar, T_decal_cam


def compose_net_decalibration(
    T_decal_lidar: Transform, T_decal_cam: Transform, T_gt: Transform
) -> Transform:
    """
    Compose independent camera- and LiDAR-side decalibrations into the single
    net error the model predicts, given the ground-truth extrinsic ``T_gt``.

    T_init = T_decal_cam.inverse() @ T_gt @ T_decal_lidar
    T_decal_net = T_init @ T_gt.inverse()

    With an identity camera draw, the LiDAR-local perturbation is conjugated
    into the camera frame by T_gt. This differs from ErrorGenerator's legacy
    camera-frame relative perturbation when T_gt is nonidentity.
    """
    T_init = T_decal_cam.inverse() @ T_gt @ T_decal_lidar
    return T_init @ T_gt.inverse()


class ErrorGenerator6D:
    """
    Per-axis decalibration ranges (lists of three values, one per axis).

    Args:
        r_range: [rx_deg, ry_deg, rz_deg]
        t_range: [tx_cm, ty_cm, tz_cm]
    """

    def __init__(self, r_range: list[float], t_range: list[float]) -> None:
        self._r_rad = [math.radians(r) for r in r_range]
        self._t_m = [t / 100.0 for t in t_range]

    def __call__(self) -> Transform:
        rot = torch.tensor([torch.empty(1).uniform_(-r, r).item() for r in self._r_rad])
        trans = torch.tensor([torch.empty(1).uniform_(-t, t).item() for t in self._t_m])
        T = euler_to_transform_matrix(trans, rot).cpu().numpy()
        return Transform(T)
