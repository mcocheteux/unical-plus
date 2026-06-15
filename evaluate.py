"""
Standalone evaluation on the test split.

Usage:
    python evaluate.py data_dir=/path/to/kitti_raw ckpt=logs/checkpoints/last.ckpt
"""
from __future__ import annotations

import os

import hydra
import pytorch_lightning as L
from hydra.utils import instantiate
from omegaconf import DictConfig

os.sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))



@hydra.main(config_path="configs", config_name="train", version_base="1.3")
def main(cfg: DictConfig) -> None:
    ckpt: str = cfg.get("ckpt", None)
    assert ckpt, "Provide a checkpoint path: ckpt=/path/to/model.ckpt"

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
