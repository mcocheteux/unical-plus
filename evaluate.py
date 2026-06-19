"""
Standalone evaluation on the test split.

Usage:
    python evaluate.py data_dir=/path/to/kitti_raw +ckpt=logs/checkpoints/last.ckpt
"""

from __future__ import annotations

import hydra
import pytorch_lightning as L
import torch
from hydra.utils import instantiate
from omegaconf import DictConfig

torch.set_float32_matmul_precision("high")


@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ckpt: str = cfg.get("ckpt", None)
    assert ckpt, "Provide a checkpoint path: +ckpt=/path/to/model.ckpt"

    L.seed_everything(cfg.seed, workers=True)

    datamodule = instantiate(cfg.data)
    model = instantiate(cfg.model)

    trainer = L.Trainer(
        accelerator=cfg.trainer.accelerator,
        devices=1,
        logger=False,
    )
    trainer.test(model, datamodule=datamodule, ckpt_path=ckpt)


if __name__ == "__main__":
    main()
