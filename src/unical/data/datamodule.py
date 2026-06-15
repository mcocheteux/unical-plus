"""
Lightning DataModule wrapping the KITTI dataset.
"""
from __future__ import annotations

from typing import Any

import pytorch_lightning as L
from torch.utils.data import DataLoader

from unical.data.dataset import KittiDataset


class KittiDataModule(L.LightningDataModule):
    """
    Manages train / val / test splits of the KITTI raw dataset.

    Args:
        data_dir:     Path to KITTI raw root.
        splits:       Dict with keys "train", "val", "test", each a
                      List[Tuple[date_str, List[drive_id]]].
        preprocessor: Configured DataPreprocessor instance.
        decalibrator: Configured ErrorGenerator instance.
        batch_size:   Samples per GPU.
        num_workers:  DataLoader workers.
        pin_memory:   Pin memory for GPU transfers.
    """

    def __init__(
        self,
        data_dir:     str,
        splits:       dict[str, Any],
        preprocessor: Any,
        decalibrator: Any,
        batch_size:   int  = 8,
        num_workers:  int  = 4,
        pin_memory:   bool = True,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["preprocessor", "decalibrator", "splits"])
        self._splits       = splits
        self._preprocessor = preprocessor
        self._decalibrator = decalibrator

    # ------------------------------------------------------------------

    def setup(self, stage: str | None = None) -> None:
        def _make(key: str) -> KittiDataset:
            # val/test use deterministic (seeded-per-index) decalibrations so their
            # metrics are stable and reproducible across epochs and runs.
            return KittiDataset(
                data_dir      = self.hparams.data_dir,
                split         = self._splits[key],
                preprocessor  = self._preprocessor,
                decalibrator  = self._decalibrator,
                deterministic = key != "train",
            )
        self.train_ds = _make("train")
        self.val_ds   = _make("val")
        self.test_ds  = _make("test")

    def _loader(self, ds: KittiDataset, shuffle: bool) -> DataLoader:
        return DataLoader(
            ds,
            batch_size  = self.hparams.batch_size,
            num_workers = self.hparams.num_workers,
            pin_memory  = self.hparams.pin_memory,
            shuffle     = shuffle,
            collate_fn  = KittiDataset.collate,
            persistent_workers = self.hparams.num_workers > 0,
        )

    def train_dataloader(self) -> DataLoader:
        return self._loader(self.train_ds, shuffle=True)

    def val_dataloader(self) -> DataLoader:
        return self._loader(self.val_ds, shuffle=False)

    def test_dataloader(self) -> DataLoader:
        return self._loader(self.test_ds, shuffle=False)
