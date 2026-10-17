"""Local CUDA training checks for the full temporal model and AMP geometry."""

from __future__ import annotations

import numpy as np
import pytest
import torch

from unical.data.dataset import Batch
from unical.losses.combined import CombinedLoss
from unical.losses.regression import RegressionLoss
from unical.losses.spatial import SpatialLoss
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead
from unical.models.module import UniCal
from unical.models.temporal import TemporalFusion
from unical.utils.transform import Transform


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires a local CUDA GPU")
@pytest.mark.parametrize(
    "fusion_type,T", [("none", 1), ("none", 3), ("gru", 3), ("transformer", 3)]
)
@pytest.mark.parametrize("amp", [False, True], ids=["fp32", "bf16"])
def test_cuda_optimizer_step_and_checkpoint(tmp_path, fusion_type, T, amp):
    """Every model block receives finite gradients and survives a checkpoint reload."""
    torch.manual_seed(42)
    device = torch.device("cuda:0")
    model = UniCal(
        MobileViTBackbone(image_size=64, pretrained=None),
        SplitRegressionHead(in_features=640, common_hidden=[], trans_hidden=[32], rot_hidden=[32]),
        CombinedLoss(RegressionLoss(), SpatialLoss()),
        temporal=TemporalFusion(
            fusion_type=fusion_type,
            gru_hidden=32,
            transformer_layers=1,
            transformer_ff_dim=128,
        ),
    ).to(device)
    B, N = 2, 32
    target_t = torch.full((B, 3), 0.01, device=device)
    target_R = torch.eye(3, device=device).repeat(B, 1, 1)
    gt = Transform(np.eye(4))
    init = Transform.from_rotation_translation(np.eye(3), np.full(3, 0.01))
    batch = Batch(
        img=torch.randn(B, T, 3, 64, 64, device=device),
        lidar_map=torch.rand(B, T, 1, 64, 64, device=device),
        target_reg=(target_t, target_R),
        pcl=[torch.rand(B, N, 4, device=device) for _ in range(T)],
        metadata=[[{"T_gt": gt, "T_init": init} for _ in range(B)] for _ in range(T)],
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
    before = model.head.trans_head[-1].weight.detach().clone()
    with torch.autocast("cuda", dtype=torch.bfloat16, enabled=amp):
        losses, predictions, _ = model._step(batch)
    assert all(torch.isfinite(value) for value in losses.values())
    for prediction in predictions:
        np.testing.assert_allclose(
            prediction.rotation_matrix @ prediction.rotation_matrix.T,
            np.eye(3),
            atol=1e-5,
        )
    losses["loss"].backward()
    for block in [model.backbone, model.head, model.temporal]:
        grads = [p.grad for p in block.parameters() if p.requires_grad]
        if grads:
            assert all(g is not None and torch.isfinite(g).all() for g in grads)
            assert sum(g.abs().sum() for g in grads) > 0
    optimizer.step()
    assert not torch.equal(before, model.head.trans_head[-1].weight)

    model.eval()
    with torch.no_grad():
        expected = model(batch)
    checkpoint = tmp_path / "model.pt"
    torch.save(model.state_dict(), checkpoint)
    with torch.no_grad():
        model.head.trans_head[-1].weight.zero_()
    model.load_state_dict(torch.load(checkpoint, weights_only=True, map_location=device))
    with torch.no_grad():
        actual = model(batch)
    for lhs, rhs in zip(expected, actual):
        torch.testing.assert_close(lhs, rhs)
