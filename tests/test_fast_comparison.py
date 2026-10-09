"""Speed controls preserve dense inputs and calibration targets."""

import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from fast_c2l import LossPointSubset  # noqa: E402


class Samples:
    def __init__(self, temporal=False):
        points = torch.arange(80).reshape(20, 4).float()
        self.sample = {
            "pcl": [points, points + 100] if temporal else points,
            "img": torch.ones(3, 32, 32),
            "lidar_map": torch.ones(1, 32, 32),
            "trans": torch.ones(3),
            "metadata": {"frame_index": 7},
        }

    def __len__(self):
        return 1

    def __getitem__(self, index):
        return self.sample


def test_loss_point_cap_retains_inputs_targets_and_scan_coverage():
    original = Samples()
    limited = LossPointSubset(original, 8)
    sample = limited[0]
    assert len(sample["pcl"]) == 8
    assert torch.equal(sample["pcl"][0], original.sample["pcl"][0])
    assert torch.equal(sample["pcl"][-1], original.sample["pcl"][-1])
    assert torch.equal(sample["pcl"], limited[0]["pcl"])
    for key in ["img", "lidar_map", "trans", "metadata"]:
        assert sample[key] is original.sample[key]
    assert len(original.sample["pcl"]) == 20


def test_temporal_frames_are_capped_independently_and_short_scans_kept():
    original = Samples(temporal=True)
    limited = LossPointSubset(original, 8)
    assert [len(frame) for frame in limited[0]["pcl"]] == [8, 8]
    assert torch.equal(limited[0]["pcl"][1][-1], original.sample["pcl"][1][-1])
    uncropped = LossPointSubset(original, 30)[0]
    assert all(a is b for a, b in zip(uncropped["pcl"], original.sample["pcl"]))
