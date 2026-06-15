"""
Spatial (point-cloud) loss.

Compares the 3-D positions of LiDAR points projected with the ground-truth
calibration vs. the recalibrated one (using the network's prediction).  Two
complementary terms are available:

  centroid_loss — MSE between point-cloud centroids in camera frame
  pcl_loss      — MSE between point-to-point distances (after truncation)

Reference: https://ieeexplore.ieee.org/document/9599702
"""
from __future__ import annotations

import torch
import torch.nn as nn

from unical.data.dataset import Batch
from unical.utils.transform import Transform


class SpatialLoss(nn.Module):
    """
    Args:
        centroid_weight: Weight on the centroid MSE term.
        pcl_weight:      Weight on the full point-cloud MSE term.
    """

    def __init__(self, centroid_weight: float = 1.0, pcl_weight: float = 1.0) -> None:
        super().__init__()
        self.centroid_weight = centroid_weight
        self.pcl_weight      = pcl_weight
        self._mse = nn.MSELoss()

    def forward(
        self,
        pred:  tuple[torch.Tensor, torch.Tensor],
        batch: Batch,
    ) -> dict[str, torch.Tensor]:
        pred_t, pred_r = pred
        device = pred_t.device
        B = pred_t.shape[0]

        pts_gt_list:   list[torch.Tensor] = []
        pts_pred_list: list[torch.Tensor] = []
        min_pts = float("inf")

        for i in range(B):
            # Reconstruct predicted recalibration in camera frame
            T_pred   = Transform.from_euler(pred_t[i], pred_r[i])
            T_fix    = T_pred.inverse()
            T_init   = batch.metadata[i]["T_init"]   # Transform (numpy)
            T_recalib = T_fix @ T_init                # Transform

            # Raw scan: keep only points with intensity > 0 (filters padding zeros)
            scan = batch.pcl[i]                       # (N_max, 4) on device
            mask = scan[:, 3] > 0
            pts  = scan[mask, :4].clone()             # (n, 4)
            pts[:, 3] = 1.0                           # homogenise

            # Ground-truth projection: T_gt @ pts
            T_gt_torch = batch.metadata[i]["T_gt"].to_torch(device)    # (4, 4)
            pts_gt     = (T_gt_torch @ pts.T).T[:, :3].unsqueeze(0)    # (1, n, 3)

            # Predicted recalibrated projection
            T_rec_torch = T_recalib.to_torch(device)                   # (4, 4)
            pts_rec     = (T_rec_torch @ pts.T).T[:, :3].unsqueeze(0)  # (1, n, 3)

            min_pts = min(min_pts, pts.shape[0])
            pts_gt_list.append(pts_gt)
            pts_pred_list.append(pts_rec)

        min_pts = int(min_pts)
        pts_gt   = torch.cat([p[:, :min_pts] for p in pts_gt_list],   dim=0)  # (B, n, 3)
        pts_pred = torch.cat([p[:, :min_pts] for p in pts_pred_list], dim=0)  # (B, n, 3)

        c_gt   = pts_gt.mean(dim=1)    # (B, 3)
        c_pred = pts_pred.mean(dim=1)  # (B, 3)

        centroid_loss = self._mse(c_pred, c_gt)   * self.centroid_weight
        pcl_loss      = self._mse(pts_pred, pts_gt) * self.pcl_weight

        return {
            "loss/spatial_centroid": centroid_loss,
            "loss/spatial_pcl":      pcl_loss,
        }
