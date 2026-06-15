"""Smoke tests for the backbone / head / module wiring (no pretrained download)."""
import torch

from unical.data.dataset import Batch
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead


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
    head = SplitRegressionHead(in_features=640, common_hidden=[],
                               trans_hidden=[64], rot_hidden=[64], rot_dim=6)
    trans, rot = head(torch.randn(2, 640))
    assert trans.shape == (2, 3)
    assert rot.shape == (2, 6)


def test_stem_inflation_channel_count():
    bb = MobileViTBackbone(img_channels=3, lidar_channels=1, image_size=64, pretrained=None)
    # from-scratch path already builds a 4-channel stem
    assert bb.model.conv_stem.convolution.in_channels == 4
