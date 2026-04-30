"""
Training entry point.

Usage:
    python train.py data_dir=/path/to/kitti_raw
    python train.py data_dir=/path/to/kitti_raw experiment=debug trainer.max_epochs=10
    python train.py data_dir=/path/to/kitti_raw trainer.devices=1 trainer.accelerator=mps
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning.callbacks import LearningRateMonitor, ModelCheckpoint, RichProgressBar

os.sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    L.seed_everything(cfg.seed, workers=True)

    # ── DataModule ────────────────────────────────────────────────────
    datamodule = instantiate(cfg.data)

    # ── Model (Hydra instantiates backbone / head / loss recursively) ─
    model = instantiate(cfg.model)

    # ── Callbacks ─────────────────────────────────────────────────────
    callbacks: list[L.Callback] = [
        ModelCheckpoint(
            dirpath   = os.path.join(cfg.log_dir, "checkpoints"),
            filename  = "unical-{epoch:03d}-{val/loss:.4f}",
            monitor   = "val/loss",
            mode      = "min",
            save_last = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        RichProgressBar(),
    ]

    # ── Trainer ───────────────────────────────────────────────────────
    trainer = L.Trainer(
        **OmegaConf.to_container(cfg.trainer, resolve=True),
        callbacks        = callbacks,
        default_root_dir = cfg.log_dir,
    )

    trainer.fit(model, datamodule=datamodule)
    trainer.test(model, datamodule=datamodule, ckpt_path="best")


if __name__ == "__main__":
    main()
