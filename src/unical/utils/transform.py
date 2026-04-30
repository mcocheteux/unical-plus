"""
Rigid-body transform utilities.

Transform wraps a 4x4 numpy homogeneous matrix and provides conversions to/from
Euler angles, quaternions, and rotation matrices.  All heavy lifting stays in
numpy so the class can live safely in dataset workers and metadata dicts without
touching any specific device.
"""
from __future__ import annotations

import math
from typing import Tuple

import numpy as np
import torch


# ---------------------------------------------------------------------------
# Axis-sequence tables (Shoemake / robotics-toolbox convention)
# ---------------------------------------------------------------------------
_NEXT_AXIS = [1, 2, 0, 1]
_AXES2TUPLE: dict[str, tuple[int, int, int, int]] = {
    "sxyz": (0, 0, 0, 0), "sxyx": (0, 0, 1, 0), "sxzy": (0, 1, 0, 0),
    "sxzx": (0, 1, 1, 0), "syzx": (1, 0, 0, 0), "syzy": (1, 0, 1, 0),
    "syxz": (1, 1, 0, 0), "syxy": (1, 1, 1, 0), "szxy": (2, 0, 0, 0),
    "szxz": (2, 0, 1, 0), "szyx": (2, 1, 0, 0), "szyz": (2, 1, 1, 0),
    "rzyx": (0, 0, 0, 1), "rxyx": (0, 0, 1, 1), "ryzx": (0, 1, 0, 1),
    "rxzx": (0, 1, 1, 1), "rxzy": (1, 0, 0, 1), "ryzy": (1, 0, 1, 1),
    "rzxy": (1, 1, 0, 1), "ryxy": (1, 1, 1, 1), "ryxz": (2, 0, 0, 1),
    "rzxz": (2, 0, 1, 1), "rxyz": (2, 1, 0, 1), "rzyz": (2, 1, 1, 1),
}
_EPS = np.finfo(float).eps * 4.0


# ---------------------------------------------------------------------------
# Low-level rotation helpers (numpy)
# ---------------------------------------------------------------------------

def euler_to_matrix_np(ai: float, aj: float, ak: float, axes: str = "sxyz") -> np.ndarray:
    """Return 4x4 homogeneous rotation matrix from Euler angles."""
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes]
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    if frame:
        ai, ak = ak, ai
    if parity:
        ai, aj, ak = -ai, -aj, -ak
    si, sj, sk = math.sin(ai), math.sin(aj), math.sin(ak)
    ci, cj, ck = math.cos(ai), math.cos(aj), math.cos(ak)
    cc, cs = ci * ck, ci * sk
    sc, ss = si * ck, si * sk
    T = np.eye(4)
    if repetition:
        T[i, i] = cj;    T[i, j] = sj * si; T[i, k] = sj * ci
        T[j, i] = sj * sk; T[j, j] = -cj * ss + cc; T[j, k] = -cj * cs - sc
        T[k, i] = -sj * ck; T[k, j] = cj * sc + cs; T[k, k] = cj * cc - ss
    else:
        T[i, i] = cj * ck; T[i, j] = sj * sc - cs; T[i, k] = sj * cc + ss
        T[j, i] = cj * sk; T[j, j] = sj * ss + cc; T[j, k] = sj * cs - sc
        T[k, i] = -sj;     T[k, j] = cj * si;       T[k, k] = cj * ci
    return T


def matrix_to_euler_np(T: np.ndarray, axes: str = "sxyz") -> tuple[float, float, float]:
    """Return Euler angles from a 4x4 (or 3x3) rotation matrix."""
    firstaxis, parity, repetition, frame = _AXES2TUPLE[axes.lower()]
    i = firstaxis
    j = _NEXT_AXIS[i + parity]
    k = _NEXT_AXIS[i - parity + 1]
    M = np.asarray(T, dtype=np.float64)[:3, :3]
    if repetition:
        sy = math.sqrt(M[i, j] ** 2 + M[i, k] ** 2)
        if sy > _EPS:
            ax = math.atan2(M[i, j], M[i, k])
            ay = math.atan2(sy, M[i, i])
            az = math.atan2(M[j, i], -M[k, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(sy, M[i, i])
            az = 0.0
    else:
        cy = math.sqrt(M[i, i] ** 2 + M[j, i] ** 2)
        if cy > _EPS:
            ax = math.atan2(M[k, j], M[k, k])
            ay = math.atan2(-M[k, i], cy)
            az = math.atan2(M[j, i], M[i, i])
        else:
            ax = math.atan2(-M[j, k], M[j, j])
            ay = math.atan2(-M[k, i], cy)
            az = 0.0
    if parity:
        ax, ay, az = -ax, -ay, -az
    if frame:
        ax, az = az, ax
    return ax, ay, az


def matrix_to_quaternion_np(T: np.ndarray) -> np.ndarray:
    """Return quaternion [w, x, y, z] from a 4x4 homogeneous matrix."""
    M = np.array(T, dtype=np.float64, copy=False)[:4, :4]
    m00, m01, m02 = M[0, 0], M[0, 1], M[0, 2]
    m10, m11, m12 = M[1, 0], M[1, 1], M[1, 2]
    m20, m21, m22 = M[2, 0], M[2, 1], M[2, 2]
    K = np.array([
        [m00 - m11 - m22, m01 + m10,       m02 + m20,       m21 - m12],
        [m01 + m10,       m11 - m00 - m22, m12 + m21,       m02 - m20],
        [m02 + m20,       m12 + m21,       m22 - m00 - m11, m10 - m01],
        [m21 - m12,       m02 - m20,       m10 - m01,       m00 + m11 + m22],
    ], dtype=np.float64) / 3.0
    w, V = np.linalg.eigh(K)
    q = V[[3, 0, 1, 2], np.argmax(w)]
    if q[0] < 0.0:
        q = -q
    return q


def quaternion_to_matrix_np(q: np.ndarray) -> np.ndarray:
    """Return 4x4 homogeneous matrix from quaternion [w, x, y, z]."""
    q = np.array(q, dtype=np.float64, copy=True)
    n = np.dot(q, q)
    if n < _EPS:
        return np.eye(4)
    q *= math.sqrt(2.0 / n)
    q = np.outer(q, q)
    return np.array([
        [1.0 - q[2, 2] - q[3, 3], q[1, 2] - q[3, 0], q[1, 3] + q[2, 0], 0.0],
        [q[1, 2] + q[3, 0], 1.0 - q[1, 1] - q[3, 3], q[2, 3] - q[1, 0], 0.0],
        [q[1, 3] - q[2, 0], q[2, 3] + q[1, 0], 1.0 - q[1, 1] - q[2, 2], 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ])


# ---------------------------------------------------------------------------
# Differentiable (PyTorch) helpers — used inside the loss / training graph
# ---------------------------------------------------------------------------

def _axis_angle_rotation(axis: str, angle: torch.Tensor) -> torch.Tensor:
    cos = torch.cos(angle)
    sin = torch.sin(angle)
    one = torch.ones_like(angle)
    zero = torch.zeros_like(angle)
    if axis == "X":
        flat = (one, zero, zero, zero, cos, -sin, zero, sin, cos)
    elif axis == "Y":
        flat = (cos, zero, sin, zero, one, zero, -sin, zero, cos)
    elif axis == "Z":
        flat = (cos, -sin, zero, sin, cos, zero, zero, zero, one)
    else:
        raise ValueError(f"axis must be X, Y or Z, got {axis!r}")
    return torch.stack(flat, -1).reshape(angle.shape + (3, 3))


def euler_to_rotation_matrix(euler: torch.Tensor, convention: str = "XYZ") -> torch.Tensor:
    """
    Differentiable Euler → 3x3 rotation matrix.

    Args:
        euler: (..., 3) radians
        convention: 3-char string like "XYZ"
    """
    matrices = [
        _axis_angle_rotation(c, e)
        for c, e in zip(convention, torch.unbind(euler, -1))
    ]
    return torch.matmul(torch.matmul(matrices[0], matrices[1]), matrices[2])


def euler_to_transform_matrix(trans: torch.Tensor, rot: torch.Tensor) -> torch.Tensor:
    """
    Build a 4x4 homogeneous transform from translation (3,) and Euler angles (3,).
    Stays on the device of the inputs.
    """
    R = euler_to_rotation_matrix(rot)      # (3, 3)
    device, dtype = trans.device, trans.dtype
    T = torch.zeros(4, 4, device=device, dtype=dtype)
    T[:3, :3] = R
    T[:3, 3] = trans.squeeze()
    T[3, 3] = 1.0
    return T


# ---------------------------------------------------------------------------
# Transform class
# ---------------------------------------------------------------------------

class Transform:
    """
    Rigid-body transform backed by a numpy 4×4 homogeneous matrix.

    Intentionally numpy-only so it can be stored in dataset metadata dicts
    and passed across DataLoader workers safely.  Conversion to torch tensors
    happens at the boundary (dataset collate / loss functions).
    """

    def __init__(self, matrix: np.ndarray) -> None:
        assert matrix.shape == (4, 4), f"Expected (4,4), got {matrix.shape}"
        self._T = matrix.astype(np.float32)

    # ------------------------------------------------------------------
    # Constructors
    # ------------------------------------------------------------------

    @classmethod
    def from_matrix(cls, matrix: np.ndarray) -> "Transform":
        return cls(matrix)

    @classmethod
    def from_rotation_translation(cls, R: np.ndarray, t: np.ndarray) -> "Transform":
        assert R.shape == (3, 3), f"Expected (3,3), got {R.shape}"
        assert t.shape == (3,),   f"Expected (3,), got {t.shape}"
        T = np.eye(4, dtype=np.float32)
        T[:3, :3] = R
        T[:3, 3] = t
        return cls(T)

    @classmethod
    def from_euler(cls, trans: torch.Tensor | np.ndarray, rot: torch.Tensor | np.ndarray) -> "Transform":
        """Create from translation (3,) and Euler angles (3,) — supports tensors on any device."""
        if isinstance(trans, torch.Tensor):
            trans_np = trans.detach().cpu().float().numpy().squeeze()
        else:
            trans_np = np.asarray(trans, dtype=np.float32).squeeze()
        if isinstance(rot, torch.Tensor):
            rot_np = rot.detach().cpu().float().numpy().squeeze()
        else:
            rot_np = np.asarray(rot, dtype=np.float32).squeeze()
        ai, aj, ak = float(rot_np[0]), float(rot_np[1]), float(rot_np[2])
        T = euler_to_matrix_np(ai, aj, ak)
        T[:3, 3] = trans_np
        return cls(T.astype(np.float32))

    @classmethod
    def from_quaternion(cls, trans: np.ndarray, quat: np.ndarray) -> "Transform":
        """Create from translation (3,) and quaternion [w,x,y,z] (4,)."""
        T = quaternion_to_matrix_np(quat)
        T[:3, 3] = trans
        return cls(T.astype(np.float32))

    # ------------------------------------------------------------------
    # Operators
    # ------------------------------------------------------------------

    def __matmul__(self, other: "Transform") -> "Transform":
        return Transform(self._T @ other._T)

    def inverse(self) -> "Transform":
        R = self._T[:3, :3].T
        t = self._T[:3, 3]
        inv = np.eye(4, dtype=np.float32)
        inv[:3, :3] = R
        inv[:3, 3] = -(R @ t)
        return Transform(inv)

    def __str__(self) -> str:
        return f"Transform(t={self.translation}, euler={np.degrees(self.euler)}°)"

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def matrix(self) -> np.ndarray:
        """4×4 homogeneous matrix (float32 numpy)."""
        return self._T

    # Alias used by the original codebase — kept for compatibility
    @property
    def T(self) -> np.ndarray:
        return self._T

    @property
    def translation(self) -> np.ndarray:
        return self._T[:3, 3].copy()

    @property
    def rotation_matrix(self) -> np.ndarray:
        return self._T[:3, :3].copy()

    @property
    def euler(self) -> np.ndarray:
        ax, ay, az = matrix_to_euler_np(self._T)
        return np.array([ax, ay, az], dtype=np.float32)

    @property
    def quaternion(self) -> np.ndarray:
        return matrix_to_quaternion_np(self._T).astype(np.float32)

    # ------------------------------------------------------------------
    # Decompose to target representation
    # ------------------------------------------------------------------

    def to_euler_components(self) -> Tuple[np.ndarray, np.ndarray]:
        """Return (translation (3,), euler_angles (3,)) both float32."""
        return self.translation, self.euler

    def to_torch(self, device: torch.device | str = "cpu") -> torch.Tensor:
        """Return 4×4 matrix as a float32 torch tensor on *device*."""
        return torch.from_numpy(self._T).to(device=device, dtype=torch.float32)
