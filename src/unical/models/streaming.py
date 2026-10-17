"""Reuse per-frame encoder features during causal sequence inference."""

from __future__ import annotations

from collections import deque
from collections.abc import Hashable
from typing import TYPE_CHECKING

import torch

from unical.data.dataset import Batch

if TYPE_CHECKING:
    from unical.models.module import UniCal


def _tensor_state(tensor: torch.Tensor) -> tuple:
    try:
        version = tensor._version
    except RuntimeError:  # Tensors created inside inference_mode have no version counter.
        version = None
    return id(tensor), version, tensor.data_ptr(), tensor.device, tensor.dtype, tensor.shape


def _autocast_state(device_type: str) -> tuple:
    if hasattr(torch, "get_autocast_dtype"):
        return torch.is_autocast_enabled(device_type), torch.get_autocast_dtype(device_type)
    # PyTorch 2.2 uses separate CPU and CUDA autocast accessors.
    if device_type == "cpu":
        return torch.is_autocast_cpu_enabled(), torch.get_autocast_cpu_dtype()
    return torch.is_autocast_enabled(), torch.get_autocast_gpu_dtype()


class StreamingCalibrator:
    """Encode each new observation once, then fuse its bounded causal history.

    ``step`` accepts the usual Batch with T=1, and returns the model's translation
    and 6-D rotation predictions once ``sequence_length`` observations are ready.
    The model and every child module must be in eval mode, with fixed weights.

    Use one instance per stream or consistently ordered batch of streams. Change
    ``context_id`` whenever stream identity, camera intrinsics, or the projection
    calibration changes. Gaps/repeated frames and observed model/precision changes
    clear history automatically. Call ``reset`` after external state changes that
    do not update tensor version counters, including inference-mode mutations.
    """

    def __init__(self, model: UniCal, sequence_length: int = 3, frame_stride: int = 1):
        if sequence_length < 1 or frame_stride < 1:
            raise ValueError("Sequence length and frame stride must be positive")
        if (
            model.temporal.fusion_type == "transformer"
            and sequence_length > model.temporal.pos_embedding.shape[1]
        ):
            raise ValueError("Sequence length exceeds Transformer positional capacity")
        self.model = model
        self.sequence_length = sequence_length
        self.frame_stride = frame_stride
        self._features: deque[torch.Tensor] = deque(maxlen=sequence_length)
        self.reset()

    @property
    def history_length(self) -> int:
        """Number of retained frame features, bounded by sequence_length."""
        return len(self._features)

    def reset(self) -> None:
        """Release retained features and begin a fresh context."""
        self._features.clear()
        self._context_id: Hashable | None = None
        self._frame_index: int | None = None
        self._signature: tuple | None = None

    def _state_signature(self, batch: Batch) -> tuple:
        tensors = (*self.model.parameters(), *self.model.buffers())
        return (
            tuple(_tensor_state(tensor) for tensor in tensors),
            batch.img.shape,
            batch.img.device,
            batch.img.dtype,
            batch.lidar_map.shape,
            batch.lidar_map.device,
            batch.lidar_map.dtype,
            _autocast_state(batch.img.device.type),
            torch.get_float32_matmul_precision(),
            torch.backends.cuda.matmul.allow_tf32,
            torch.backends.cudnn.allow_tf32,
        )

    @torch.no_grad()
    def step(
        self, batch: Batch, *, context_id: Hashable, frame_index: int
    ) -> tuple[torch.Tensor, torch.Tensor] | None:
        """Return a prediction for the newest frame, or None during history warmup."""
        if any(module.training for module in self.model.modules()):
            self.reset()
            raise ValueError("Streaming inference requires every model module in eval mode")
        if (
            batch.img.ndim != 5
            or batch.lidar_map.ndim != 5
            or batch.img.shape[1] != 1
            or batch.lidar_map.shape[1] != 1
            or batch.img.shape[0] != batch.lidar_map.shape[0]
            or batch.img.shape[-2:] != batch.lidar_map.shape[-2:]
        ):
            raise ValueError(
                "Streaming expects matching image/LiDAR batches with exactly one frame"
            )
        hash(context_id)  # Require an immutable, hashable context identity.
        signature = self._state_signature(batch)
        if (
            context_id != self._context_id
            or signature != self._signature
            or (
                self._frame_index is not None
                and frame_index != self._frame_index + self.frame_stride
            )
        ):
            self.reset()
        flat_batch = batch._replace(img=batch.img[:, 0], lidar_map=batch.lidar_map[:, 0])
        feature = self.model.backbone(flat_batch).detach()
        if self._features and (
            feature.shape != self._features[-1].shape
            or feature.dtype != self._features[-1].dtype
            or feature.device != self._features[-1].device
        ):
            self.reset()
        self._features.append(feature)
        self._context_id = context_id
        self._frame_index = frame_index
        self._signature = signature
        if len(self._features) < self.sequence_length:
            return None
        features = torch.stack(tuple(self._features), dim=1)
        return self.model.head(self.model.temporal(features))
