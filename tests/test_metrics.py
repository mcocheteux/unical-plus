"""Calibration metrics must report geodesic rotation in degrees."""

from __future__ import annotations

import numpy as np
import pytest

from unical.utils.metrics import CalibMetrics
from unical.utils.transform import Transform


@pytest.mark.parametrize("angle", [0.0, 30.0, 90.0, 180.0])
def test_global_rotation_is_geodesic_angle(angle):
    metrics = CalibMetrics()
    prediction = Transform.from_euler(np.zeros(3), np.radians([0.0, 0.0, angle]))
    metrics.add(prediction, Transform(np.eye(4)))
    result = metrics.rotation_metrics()
    assert result["rot/global/MAE"] == pytest.approx(angle, abs=1e-4)
    assert result["rot/global/STD"] == 0.0
    assert result["rot/yaw/MAE"] == pytest.approx(angle, abs=1e-4)
