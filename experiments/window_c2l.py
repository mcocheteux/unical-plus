"""Expose published window anchors through exact main's single-frame Batch contract."""

from pathlib import Path
from typing import Any

from unical.data.kitti_c2l_dataset import KittiC2LDataModule, KittiC2LDataset


def expand_anchors(rows: list[dict], context: int) -> list[dict]:
    """Match branch T=1/T=3 anchors while retaining each published window's target."""
    if context < 1:
        raise ValueError("Minimum context must be positive")
    return [
        row
        | {
            "sample_id": row["window_id"],
            "frame_index": row["window_start_frame"] + offset,
            "raw_frame_index": row["raw_frame_start_index"] + offset,
        }
        for row in rows
        for offset in range(context - 1, row["window_length"])
    ]


class MainWindowDataset(KittiC2LDataset):
    """Use main's original image preparation, target construction and collator."""

    def __init__(self, *args: Any, minimum_context_length: int = 3, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self._rows = expand_anchors(self._rows, minimum_context_length)

    def __getitem__(self, index: int) -> dict:
        sample = super().__getitem__(index)
        if sample["img"].ndim != 3:
            raise RuntimeError("MainWindowDataset requires the unchanged main checkout")
        sample["metadata"]["window_id"] = self._rows[index]["window_id"]
        return sample


class MainWindowDataModule(KittiC2LDataModule):
    def __init__(self, minimum_context_length: int = 3, **kwargs: Any) -> None:
        if kwargs.get("stages") is not None:
            kwargs["stages"] = list(kwargs["stages"])
        super().__init__(**kwargs)
        self.minimum_context_length = minimum_context_length

    def setup(self, stage: str | None = None) -> None:
        directory = Path(self.hparams.data_dir)
        validation = {self.hparams.val_sequence}
        common = dict(
            kitti_raw_root=self.hparams.kitti_raw_root,
            preprocessor=self._preprocessor,
            stages=self._stage_set,
            minimum_context_length=self.minimum_context_length,
        )
        self.train_ds = MainWindowDataset(
            directory / "windows_train.parquet", exclude_sequences=validation, **common
        )
        self.val_ds = MainWindowDataset(
            directory / "windows_train.parquet", include_sequences=validation, **common
        )
        self.test_ds = MainWindowDataset(directory / "windows_test.parquet", **common)
