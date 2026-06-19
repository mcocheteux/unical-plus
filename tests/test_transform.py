"""Unit tests for rotation / transform utilities."""

import numpy as np
import torch

from unical.utils.transform import (
    Transform,
    build_transform_matrix,
    euler_to_transform_matrix,
    matrix_to_rotation_6d,
    rotation_6d_to_matrix,
)


def _is_rotation(R: torch.Tensor, atol: float = 1e-5) -> bool:
    eye = torch.eye(3, dtype=R.dtype)
    rrt = R @ R.transpose(-1, -2)
    return torch.allclose(rrt, eye.expand_as(rrt), atol=atol) and torch.allclose(
        torch.det(R), torch.ones(R.shape[:-2]), atol=atol
    )


def test_rotation_6d_to_matrix_is_orthonormal():
    d6 = torch.randn(8, 6)
    R = rotation_6d_to_matrix(d6)
    assert R.shape == (8, 3, 3)
    assert _is_rotation(R)


def test_6d_matrix_roundtrip():
    # start from a genuine rotation, go to 6d and back
    d6 = torch.randn(4, 6)
    R = rotation_6d_to_matrix(d6)
    R2 = rotation_6d_to_matrix(matrix_to_rotation_6d(R))
    assert torch.allclose(R, R2, atol=1e-5)


def test_build_transform_matrix():
    trans = torch.tensor([[1.0, 2.0, 3.0]])
    R = rotation_6d_to_matrix(torch.randn(1, 6))
    T = build_transform_matrix(trans, R)
    assert T.shape == (1, 4, 4)
    assert torch.allclose(T[0, :3, :3], R[0])
    assert torch.allclose(T[0, :3, 3], trans[0])
    assert torch.allclose(T[0, 3], torch.tensor([0.0, 0.0, 0.0, 1.0]))


def test_rotation_6d_gradient_flows():
    d6 = torch.randn(2, 6, requires_grad=True)
    R = rotation_6d_to_matrix(d6)
    R.sum().backward()
    assert d6.grad is not None and torch.isfinite(d6.grad).all()


def test_euler_transform_roundtrip():
    trans = np.array([0.1, -0.2, 0.05], dtype=np.float32)
    euler = np.array([0.01, -0.02, 0.03], dtype=np.float32)
    T = Transform.from_euler(trans, euler)
    assert np.allclose(T.translation, trans, atol=1e-5)
    assert np.allclose(T.euler, euler, atol=1e-5)


def test_euler_to_transform_matrix_matches_numpy():
    rot = torch.tensor([0.01, -0.02, 0.03])
    trans = torch.tensor([0.1, 0.2, 0.3])
    T = euler_to_transform_matrix(trans, rot)
    assert T.shape == (4, 4)
    assert torch.allclose(T[:3, 3], trans, atol=1e-6)
