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
        self._t_m   = t_range / 100.0          # cm → m

    def __call__(self, generator: torch.Generator | None = None) -> Transform:
        """Sample a decalibration. Pass a seeded ``generator`` for reproducibility
        (used by the val/test splits so their decalibrations are deterministic)."""
        rot   = torch.empty(3).uniform_(-self._r_rad, self._r_rad, generator=generator)
        trans = torch.empty(3).uniform_(-self._t_m,   self._t_m,   generator=generator)
        T = euler_to_transform_matrix(trans, rot).cpu().numpy()
        return Transform(T)


class ErrorGenerator6D:
    """
    Per-axis decalibration ranges (lists of three values, one per axis).

    Args:
        r_range: [rx_deg, ry_deg, rz_deg]
        t_range: [tx_cm, ty_cm, tz_cm]
    """

    def __init__(self, r_range: list[float], t_range: list[float]) -> None:
        self._r_rad = [math.radians(r) for r in r_range]
        self._t_m   = [t / 100.0 for t in t_range]

    def __call__(self) -> Transform:
        rot   = torch.tensor([torch.empty(1).uniform_(-r, r).item() for r in self._r_rad])
        trans = torch.tensor([torch.empty(1).uniform_(-t, t).item() for t in self._t_m])
        T = euler_to_transform_matrix(trans, rot).cpu().numpy()
        return Transform(T)
