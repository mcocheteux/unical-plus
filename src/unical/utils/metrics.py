"""
Calibration error metrics accumulated over an epoch.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List

import numpy as np

from unical.utils.transform import Transform, matrix_to_euler_np


@dataclass
class CalibMetrics:
    """Running storage for one epoch of predictions vs targets."""

    preds:   List[Transform] = field(default_factory=list)
    targets: List[Transform] = field(default_factory=list)

    def add(self, pred: Transform, target: Transform) -> None:
        self.preds.append(pred)
        self.targets.append(target)

    def clear(self) -> None:
        self.preds.clear()
        self.targets.clear()

    def translation_metrics(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        errors = np.array([
            np.abs(p.translation - t.translation) * 100  # → cm
            for p, t in zip(self.preds, self.targets)
        ])  # (N, 3)
        return {
            "trans/X/MAE":      float(np.mean(errors[:, 0])),
            "trans/Y/MAE":      float(np.mean(errors[:, 1])),
            "trans/Z/MAE":      float(np.mean(errors[:, 2])),
            "trans/X/STD":      float(np.std(errors[:, 0])),
            "trans/Y/STD":      float(np.std(errors[:, 1])),
            "trans/Z/STD":      float(np.std(errors[:, 2])),
            "trans/global/MAE": float(np.mean(errors)),
            "trans/global/STD": float(np.std(errors)),
        }

    def rotation_metrics(self) -> Dict[str, float]:
        if not self.preds:
            return {}
        errors = []
        for p, t in zip(self.preds, self.targets):
            # relative rotation: p^{-1} ∘ t, then decompose
            rel = p.inverse() @ t
            ax, ay, az = matrix_to_euler_np(rel.matrix)
            errors.append(np.degrees(np.abs([ax, ay, az])))
        errors = np.array(errors)  # (N, 3)
        return {
            "rot/roll/MAE":    float(np.mean(errors[:, 0])),
            "rot/pitch/MAE":   float(np.mean(errors[:, 1])),
            "rot/yaw/MAE":     float(np.mean(errors[:, 2])),
            "rot/roll/STD":    float(np.std(errors[:, 0])),
            "rot/pitch/STD":   float(np.std(errors[:, 1])),
            "rot/yaw/STD":     float(np.std(errors[:, 2])),
            "rot/global/MAE":  float(np.mean(errors)),
            "rot/global/STD":  float(np.std(errors)),
        }

    def all_metrics(self) -> Dict[str, float]:
        return {**self.translation_metrics(), **self.rotation_metrics()}
