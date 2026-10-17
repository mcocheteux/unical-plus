"""Unit tests for TemporalFusion (feature aggregation over a window of T frames)."""

import torch

from unical.models.temporal import TemporalFusion


def test_none_fusion_is_parameter_free_mean():
    fusion = TemporalFusion(feature_dim=8, fusion_type="none")
    assert sum(p.numel() for p in fusion.parameters()) == 0
    x = torch.randn(3, 4, 8)
    out = fusion(x)
    assert out.shape == (3, 8)
    assert torch.allclose(out, x.mean(dim=1))


def test_none_fusion_degenerates_at_t1():
    fusion = TemporalFusion(feature_dim=8, fusion_type="none")
    x = torch.randn(3, 1, 8)
    out = fusion(x)
    assert torch.allclose(out, x[:, 0])


def test_gru_fusion_shape_and_grad():
    fusion = TemporalFusion(feature_dim=8, fusion_type="gru", gru_hidden=8)
    x = torch.randn(3, 4, 8, requires_grad=True)
    out = fusion(x)
    assert out.shape == (3, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_fusion_shape_and_grad():
    fusion = TemporalFusion(
        feature_dim=8,
        fusion_type="transformer",
        transformer_layers=1,
        transformer_heads=2,
        transformer_ff_dim=16,
        max_seq_len=8,
    )
    x = torch.randn(3, 4, 8, requires_grad=True)
    out = fusion(x)
    assert out.shape == (3, 8)
    out.sum().backward()
    assert x.grad is not None and torch.isfinite(x.grad).all()


def test_transformer_fusion_rejects_too_long_sequence():
    fusion = TemporalFusion(feature_dim=8, fusion_type="transformer", max_seq_len=2)
    x = torch.randn(2, 4, 8)
    try:
        fusion(x)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for sequence longer than max_seq_len")


def test_unknown_fusion_type_raises():
    try:
        TemporalFusion(fusion_type="bogus")
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for unknown fusion_type")
