"""
Combined loss that sums regression and spatial terms.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unical.data.dataset import Batch
from unical.losses.regression import RegressionLoss
from unical.losses.spatial import SpatialLoss


class CombinedLoss(nn.Module):
    """
    Sums all sub-losses and exposes a single "loss" key for the trainer.

    Args:
        regression: RegressionLoss instance.
        spatial:    SpatialLoss instance (can be None to disable).
    """

    def __init__(
        self,
        regression: RegressionLoss,
        spatial: SpatialLoss | None = None,
    ) -> None:
        super().__init__()
        self.regression = regression
        self.spatial = spatial

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        losses: dict[str, torch.Tensor] = {}
        losses.update(self.regression(pred, batch))
        if self.spatial is not None:
            losses.update(self.spatial(pred, batch))
        total = sum(losses.values())
        losses["loss"] = total
        return losses
