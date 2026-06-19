"""Unit tests for the loss functions, incl. the spatial-loss gradient path."""

import numpy as np
import torch

from unical.data.dataset import Batch
from unical.losses.combined import CombinedLoss
from unical.losses.regression import RegressionLoss
from unical.losses.spatial import SpatialLoss
from unical.utils.transform import Transform, rotation_6d_to_matrix


def _fake_batch(B: int = 2, N: int = 64) -> Batch:
    trans = torch.randn(B, 3) * 0.05
    R = rotation_6d_to_matrix(torch.randn(B, 6))
    pcl = torch.randn(B, N, 4).abs() + 1.0  # forward points, intensity > 0
    metadata = []
    for _ in range(B):
        T_gt = Transform.from_rotation_translation(
            np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
        )
        T_init = Transform.from_rotation_translation(
            np.eye(3, dtype=np.float32), np.array([0.01, 0.0, 0.0], np.float32)
        )
        metadata.append({"T_gt": T_gt, "T_init": T_init})
    return Batch(
        img=torch.zeros(B, 3, 8, 8),
        lidar_map=torch.zeros(B, 1, 8, 8),
        target_reg=(trans, R),
        pcl=pcl,
        metadata=metadata,
    )


def test_regression_loss_keys_and_scalar():
    batch = _fake_batch()
    pred = (torch.randn(2, 3), torch.randn(2, 6))
    out = RegressionLoss()(pred, batch)
    assert set(out) == {"loss/reg_trans", "loss/reg_rot"}
    for v in out.values():
        assert v.ndim == 0 and torch.isfinite(v)


def test_spatial_loss_gradient_reaches_predictions():
    """Regression test for the bug where the spatial loss was detached."""
    batch = _fake_batch()
    pred_t = torch.randn(2, 3, requires_grad=True)
    pred_r6 = torch.randn(2, 6, requires_grad=True)
    out = SpatialLoss()((pred_t, pred_r6), batch)
    total = sum(out.values())
    total.backward()
    assert pred_t.grad is not None and torch.isfinite(pred_t.grad).all()
    assert pred_r6.grad is not None and torch.isfinite(pred_r6.grad).all()
    # the gradient must actually be non-zero (loss truly depends on predictions)
    assert pred_r6.grad.abs().sum() > 0


def test_spatial_loss_handles_low_precision_inputs():
    """Regression test: geometry must not choke on bf16 (AMP) predictions."""
    batch = _fake_batch()
    pred_t = torch.randn(2, 3, dtype=torch.bfloat16, requires_grad=True)
    pred_r6 = torch.randn(2, 6, dtype=torch.bfloat16, requires_grad=True)
    out = SpatialLoss()((pred_t, pred_r6), batch)
    total = sum(out.values())
    assert torch.isfinite(total)
    total.backward()
    assert pred_t.grad is not None and pred_r6.grad is not None


def test_combined_loss_has_total():
    batch = _fake_batch()
    pred = (torch.randn(2, 3, requires_grad=True), torch.randn(2, 6, requires_grad=True))
    loss = CombinedLoss(RegressionLoss(), SpatialLoss())
    out = loss(pred, batch)
    assert "loss" in out
    out["loss"].backward()
    assert pred[0].grad is not None
