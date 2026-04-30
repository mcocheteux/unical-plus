"""
Direct regression loss on predicted translation and Euler angles.
"""
from __future__ import annotations

from typing import Dict, Tuple

import torch
import torch.nn as nn

from unical.data.dataset import Batch


class RegressionLoss(nn.Module):
    """
    MSE on translation + MSE on rotation, each with its own weight.

    Args:
        trans_weight: Scalar weight on the translation MSE.
        rot_weight:   Scalar weight on the rotation MSE.
    """

    def __init__(self, trans_weight: float = 1.0, rot_weight: float = 1.0) -> None:
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight   = rot_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        pred:  Tuple[torch.Tensor, torch.Tensor],
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        pred_t, pred_r   = pred
        target_t, target_r = batch.target_reg
        t_loss = self._mse(pred_t, target_t) * self.trans_weight
        r_loss = self._mse(pred_r, target_r) * self.rot_weight
        return {"loss/reg_trans": t_loss, "loss/reg_rot": r_loss}
