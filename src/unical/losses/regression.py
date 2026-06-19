"""
Direct regression loss on predicted translation and Euler angles.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unical.data.dataset import Batch
from unical.utils.transform import rotation_6d_to_matrix


class RegressionLoss(nn.Module):
    """
    MSE on translation + (Frobenius) MSE on the rotation matrix, each weighted.

    The prediction's rotation is a continuous 6-D representation, converted to a
    rotation matrix here and compared against the target rotation matrix. Matrix
    (chordal) distance is smooth and free of Euler wrap-around issues.

    Args:
        trans_weight: Scalar weight on the translation MSE.
        rot_weight:   Scalar weight on the rotation-matrix MSE.
    """

    def __init__(self, trans_weight: float = 1.0, rot_weight: float = 1.0) -> None:
        super().__init__()
        self.trans_weight = trans_weight
        self.rot_weight = rot_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        pred_t, pred_r6 = pred
        target_t, target_R = batch.target_reg
        pred_R = rotation_6d_to_matrix(pred_r6)
        t_loss = self._mse(pred_t, target_t) * self.trans_weight
        r_loss = self._mse(pred_R, target_R) * self.rot_weight
        return {"loss/reg_trans": t_loss, "loss/reg_rot": r_loss}
