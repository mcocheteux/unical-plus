"""Shared speed controls for main and temporal comparison data loaders."""

from types import MethodType

import cv2
import torch
from torch.utils.data import DataLoader, Dataset, Subset


class LossPointSubset(Dataset):
    """Cap loss points after dense depth-map construction; retain labels and images."""

    def __init__(self, dataset, maximum: int):
        self.dataset = dataset
        self.maximum = maximum

    def __len__(self):
        return len(self.dataset)

    def _subset(self, points):
        if len(points) <= self.maximum:
            return points
        indices = torch.linspace(0, len(points) - 1, self.maximum).long()
        return points[indices]

    def __getitem__(self, index):
        sample = self.dataset[index].copy()
        points = sample["pcl"]
        sample["pcl"] = (
            [self._subset(frame) for frame in points]
            if isinstance(points, list)
            else self._subset(points)
        )
        return sample


def initialize_worker(worker_id):
    """Prevent each loader process from starting its own CPU thread pool."""
    torch.set_num_threads(1)
    cv2.setNumThreads(1)


def fast_loader(self, dataset, shuffle):
    collate = type(dataset).collate
    benchmark_samples = getattr(self, "comparison_benchmark_samples", 0)
    if shuffle and benchmark_samples:
        indices = torch.linspace(0, len(dataset) - 1, benchmark_samples).long().tolist()
        dataset = Subset(dataset, indices)
    if self.comparison_max_loss_points:
        dataset = LossPointSubset(dataset, self.comparison_max_loss_points)
    workers = self.hparams.num_workers
    return DataLoader(
        dataset,
        batch_size=self.hparams.batch_size,
        shuffle=shuffle,
        collate_fn=collate,
        num_workers=workers,
        pin_memory=self.hparams.pin_memory,
        persistent_workers=workers > 0,
        multiprocessing_context="spawn" if workers > 0 else None,
        worker_init_fn=initialize_worker if workers > 0 else None,
    )


def configure_loading(datamodule, maximum: int = 0, benchmark_samples: int = 0):
    if maximum < 0:
        raise ValueError("Loss-point cap must be nonnegative")
    datamodule.comparison_max_loss_points = maximum
    datamodule.comparison_benchmark_samples = benchmark_samples
    datamodule._loader = MethodType(fast_loader, datamodule)
