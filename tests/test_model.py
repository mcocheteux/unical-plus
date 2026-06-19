"""Smoke tests for the backbone / head / module wiring (no pretrained download)."""

import numpy as np
import torch

from unical.data.dataset import Batch
from unical.losses.combined import CombinedLoss
from unical.losses.regression import RegressionLoss
from unical.losses.spatial import SpatialLoss
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead
from unical.models.module import UniCal
from unical.utils.transform import Transform, rotation_6d_to_matrix


def _fake_batch(B: int = 2, size: int = 64) -> Batch:
    return Batch(
        img=torch.randn(B, 3, size, size),
        lidar_map=torch.randn(B, 1, size, size),
        target_reg=(torch.zeros(B, 3), torch.eye(3).expand(B, 3, 3)),
        pcl=torch.zeros(B, 4, 4),
        metadata=[{} for _ in range(B)],
    )


def test_backbone_outputs_feature_vector():
    bb = MobileViTBackbone(image_size=64, pretrained=None, spatial_head=True)
    feat = bb(_fake_batch(size=64))
    assert feat.shape == (2, bb.model.config.neck_hidden_sizes[-1])


def test_head_outputs_trans_and_6d_rotation():
    head = SplitRegressionHead(
        in_features=640, common_hidden=[], trans_hidden=[64], rot_hidden=[64], rot_dim=6
    )
    trans, rot = head(torch.randn(2, 640))
    assert trans.shape == (2, 3)
    assert rot.shape == (2, 6)


def test_stem_inflation_channel_count():
    bb = MobileViTBackbone(img_channels=3, lidar_channels=1, image_size=64, pretrained=None)
    # from-scratch path already builds a 4-channel stem
    assert bb.model.conv_stem.convolution.in_channels == 4


def test_step_runs_under_bf16_autocast():
    """End-to-end _step under bf16 AMP (guards numpy/geometry bf16 handling)."""
    bb = MobileViTBackbone(image_size=64, pretrained=None)
    feat = bb.model.config.neck_hidden_sizes[-1]
    head = SplitRegressionHead(
        in_features=feat, common_hidden=[], trans_hidden=[32], rot_hidden=[32], rot_dim=6
    )
    model = UniCal(bb, head, CombinedLoss(RegressionLoss(), SpatialLoss()))
    B, N = 2, 40
    eye, zero = np.eye(3, dtype=np.float32), np.zeros(3, dtype=np.float32)
    meta = [
        {
            "T_gt": Transform.from_rotation_translation(eye, zero),
            "T_init": Transform.from_rotation_translation(eye, np.array([0.01, 0, 0], np.float32)),
        }
        for _ in range(B)
    ]
    batch = Batch(
        img=torch.randn(B, 3, 64, 64),
        lidar_map=torch.randn(B, 1, 64, 64),
        target_reg=(torch.randn(B, 3) * 0.05, rotation_6d_to_matrix(torch.randn(B, 6))),
        pcl=torch.rand(B, N, 4) + 1.0,
        metadata=meta,
    )
    with torch.autocast(device_type="cpu", dtype=torch.bfloat16):
        losses, pred_Ts, target_Ts = model._step(batch)
    assert torch.isfinite(losses["loss"])
    assert len(pred_Ts) == B and len(target_Ts) == B
