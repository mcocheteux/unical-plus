"""
UniCal Lightning module — camera-LiDAR extrinsic calibration.

The module wires backbone + head + combined loss and handles
all Lightning hooks (train / val / test steps, metric logging).
"""

from __future__ import annotations

from typing import Any

import pytorch_lightning as L
import torch

from unical.data.dataset import Batch
from unical.losses.combined import CombinedLoss
from unical.models.backbone import MobileViTBackbone
from unical.models.head import SplitRegressionHead
from unical.models.temporal import TemporalFusion
from unical.utils.metrics import CalibMetrics
from unical.utils.transform import Transform, rotation_6d_to_matrix


class UniCal(L.LightningModule):
    """
    UniCal: joint camera-LiDAR calibration network.

    Args:
        backbone:      MobileViTBackbone instance.
        head:          SplitRegressionHead instance.
        loss:          CombinedLoss instance.
        temporal:      TemporalFusion instance, aggregating a window of T
                       per-frame backbone features into one vector before the
                       head. ``fusion_type="none"`` is a parameter-free mean
                       that is the identity at T=1, so single-frame runs are
                       unaffected.
        lr:            Adam learning rate.
        weight_decay:  Adam weight decay.
    """

    def __init__(
        self,
        backbone: MobileViTBackbone,
        head: SplitRegressionHead,
        loss: CombinedLoss,
        temporal: TemporalFusion | None = None,
        lr: float = 3e-5,
        weight_decay: float = 1e-4,
        warmup_epochs: int = 0,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["backbone", "head", "loss", "temporal"])

        self.backbone = backbone
        self.head = head
        self.loss_fn = loss
        self.temporal = temporal if temporal is not None else TemporalFusion(fusion_type="none")

        self._train_metrics = CalibMetrics()
        self._val_metrics = CalibMetrics()
        self._test_metrics = CalibMetrics()

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(self, batch: Batch) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (trans_pred (B,3), rot6d_pred (B,6))."""
        B, T = batch.img.shape[0], batch.img.shape[1]
        flat_batch = Batch(
            img=batch.img.reshape(B * T, *batch.img.shape[2:]),
            lidar_map=batch.lidar_map.reshape(B * T, *batch.lidar_map.shape[2:]),
            target_reg=batch.target_reg,
            pcl=batch.pcl,
            metadata=batch.metadata,
        )
        features = self.backbone(flat_batch).reshape(B, T, -1)  # (B, T, D)
        fused = self.temporal(features)  # (B, D)
        return self.head(fused)

    # ------------------------------------------------------------------
    # Shared step
    # ------------------------------------------------------------------

    def _step(
        self, batch: Batch
    ) -> tuple[dict[str, torch.Tensor], list[Transform], list[Transform]]:
        pred = self(batch)  # (trans, rot6d)
        losses = self.loss_fn(pred, batch)  # dict with "loss", sub-keys

        B = pred[0].shape[0]
        # .float() before .numpy(): under bf16 AMP the predictions are bfloat16,
        # which numpy cannot represent.
        pred_t = pred[0].detach().float().cpu().numpy()
        pred_R = rotation_6d_to_matrix(pred[1]).detach().float().cpu().numpy()
        tgt_t = batch.target_reg[0].detach().float().cpu().numpy()
        tgt_R = batch.target_reg[1].detach().float().cpu().numpy()
        pred_Ts = [Transform.from_rotation_translation(pred_R[i], pred_t[i]) for i in range(B)]
        target_Ts = [Transform.from_rotation_translation(tgt_R[i], tgt_t[i]) for i in range(B)]
        return losses, pred_Ts, target_Ts

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def training_step(self, batch: Batch, batch_idx: int) -> torch.Tensor:
        losses, pred_Ts, target_Ts = self._step(batch)
        B = batch.img.shape[0]
        self.log_dict(
            {f"train/{k}": v for k, v in losses.items()},
            on_step=True,
            on_epoch=False,
            prog_bar=False,
            sync_dist=True,
            batch_size=B,
        )
        return losses["loss"]

    def on_train_epoch_end(self) -> None:
        self._train_metrics.clear()

    # ------------------------------------------------------------------
    # Validation
    # ------------------------------------------------------------------

    def validation_step(self, batch: Batch, batch_idx: int) -> None:
        losses, pred_Ts, target_Ts = self._step(batch)
        B = batch.img.shape[0]
        self.log_dict(
            {f"val/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        for p, t in zip(pred_Ts, target_Ts):
            self._val_metrics.add(p, t)

    def on_validation_epoch_end(self) -> None:
        metrics = self._val_metrics.all_metrics()
        self.log_dict({f"val/{k}": v for k, v in metrics.items()}, sync_dist=True)
        self._val_metrics.clear()

    # ------------------------------------------------------------------
    # Test
    # ------------------------------------------------------------------

    def test_step(self, batch: Batch, batch_idx: int) -> None:
        losses, pred_Ts, target_Ts = self._step(batch)
        B = batch.img.shape[0]
        self.log_dict(
            {f"test/{k}": v for k, v in losses.items()},
            on_step=False,
            on_epoch=True,
            sync_dist=True,
            batch_size=B,
        )
        for p, t in zip(pred_Ts, target_Ts):
            self._test_metrics.add(p, t)

    def on_test_epoch_end(self) -> None:
        metrics = self._test_metrics.all_metrics()
        self.log_dict({f"test/{k}": v for k, v in metrics.items()}, sync_dist=True)
        self._test_metrics.clear()

    # ------------------------------------------------------------------
    # Optimiser
    # ------------------------------------------------------------------

    def configure_optimizers(self) -> dict[str, Any]:
        opt = torch.optim.AdamW(
            self.parameters(),
            lr=self.hparams.lr,
            weight_decay=self.hparams.weight_decay,
        )
        max_epochs = self.trainer.max_epochs
        warmup = min(self.hparams.warmup_epochs, max(max_epochs - 1, 0))
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(
            opt, T_max=max(max_epochs - warmup, 1), eta_min=1e-7
        )
        if warmup > 0:
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                opt, start_factor=1e-2, total_iters=warmup
            )
            scheduler: torch.optim.lr_scheduler.LRScheduler = torch.optim.lr_scheduler.SequentialLR(
                opt, schedulers=[warmup_sched, cosine], milestones=[warmup]
            )
        else:
            scheduler = cosine
        return {"optimizer": opt, "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"}}
