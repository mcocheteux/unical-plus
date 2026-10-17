"""
Spatial (point-cloud) loss.

Compares the 3-D positions of LiDAR points projected with the ground-truth
calibration vs. the recalibrated one (using the network's prediction).  Two
complementary terms are available:

  centroid_loss — MSE between point-cloud centroids in camera frame
  pcl_loss      — MSE between corresponding point positions

When the batch carries a window of T frames (``batch.pcl``/``batch.metadata``
are length-T lists), the single shared prediction is checked against every
frame's geometry and the loss is averaged over T — denser supervision from
the same decalibration draw, with T=1 reducing exactly to the original
single-frame loss.

Reference: https://ieeexplore.ieee.org/document/9599702
"""

from __future__ import annotations

import torch
import torch.nn as nn

from unical.data.dataset import Batch
from unical.utils.transform import build_transform_matrix, rotation_6d_to_matrix


class SpatialLoss(nn.Module):
    """
    Args:
        centroid_weight: Weight on the centroid MSE term.
        pcl_weight:      Weight on the full point-cloud MSE term.
    """

    def __init__(self, centroid_weight: float = 1.0, pcl_weight: float = 1.0) -> None:
        super().__init__()
        self.centroid_weight = centroid_weight
        self.pcl_weight = pcl_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        pred: tuple[torch.Tensor, torch.Tensor],
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        pred_t, pred_r6 = pred
        device = pred_t.device
        B = pred_t.shape[0]

        # Geometry must run in full precision: rigid-transform math (and the
        # ground-truth/initial extrinsics) are float32, and low-precision matmuls
        # under AMP are both inaccurate and dtype-incompatible. Disable autocast
        # and cast the predictions to float32 (gradients still flow back to AMP).
        with torch.autocast(device_type=device.type, enabled=False):
            pred_t = pred_t.float()
            pred_R = rotation_6d_to_matrix(pred_r6.float())
            R_inv = pred_R.transpose(-1, -2)
            t_inv = -torch.einsum("bij,bj->bi", R_inv, pred_t)
            T_fix = build_transform_matrix(t_inv, R_inv)

            # Keep empty scans differentiable and average each sample equally,
            # without truncating longer scans to the shortest one in the batch.
            centroid_loss = T_fix.sum() * 0.0
            pcl_loss = T_fix.sum() * 0.0
            T = len(batch.pcl)
            for t in range(T):
                for i in range(B):
                    T_init = batch.metadata[t][i]["T_init"].to_torch(device)
                    T_gt = batch.metadata[t][i]["T_gt"].to_torch(device)
                    T_recalib = T_fix[i] @ T_init

                    scan = batch.pcl[t][i].float()
                    # Collation pads with all-zero rows. Zero reflectance alone
                    # does not imply padding: it occurs in real LiDAR returns.
                    pts = scan[scan.ne(0).any(dim=-1)].clone()
                    if pts.shape[0] == 0:
                        continue
                    pts[:, 3] = 1.0
                    pts_gt = (T_gt @ pts.T).T[:, :3]
                    pts_pred = (T_recalib @ pts.T).T[:, :3]
                    centroid_loss = centroid_loss + self._mse(
                        pts_pred.mean(dim=0), pts_gt.mean(dim=0)
                    )
                    pcl_loss = pcl_loss + self._mse(pts_pred, pts_gt)

            return {
                "loss/spatial_centroid": centroid_loss / (B * T) * self.centroid_weight,
                "loss/spatial_pcl": pcl_loss / (B * T) * self.pcl_weight,
            }
