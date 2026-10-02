"""Paired research analysis must preserve targets, anchors and dependence blocks."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "paired_analysis", Path(__file__).resolve().parents[1] / "experiments/paired_analysis.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def _results(error: float) -> dict:
    return {
        "protocol": "windows",
        "stage": 5,
        "records": [
            {
                "id": str(i),
                "cluster": f"window{i // 3}",
                "target_translation_m": [0.01, 0.02, 0.03],
                "target_rotation_matrix": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
                "translation_mae_cm": error,
                "rotation_degrees": error,
                "initial_translation_mae_cm": 10.0,
                "initial_rotation_degrees": 2.0,
            }
            for i in range(12)
        ],
    }


def test_known_paired_effect_uses_four_clusters_not_twelve_independent_frames():
    result = module.compare(_results(4.0), _results(3.0), draws=200)
    assert result["clusters"] == 4
    metric = result["metrics"]["translation_mae_cm"]
    assert metric["candidate_minus_reference"] == -1.0
    assert metric["paired_95_percent_interval"] == [-1.0, -1.0]
    assert metric["supports_improvement_for_this_seed"]


@pytest.mark.parametrize("change", ["target", "signed_target", "anchor", "protocol", "duplicate"])
def test_incomparable_evaluations_are_rejected(change):
    reference, candidate = _results(4.0), _results(3.0)
    if change == "target":
        candidate["records"][0]["initial_rotation_degrees"] = 20.0
    elif change == "signed_target":
        candidate["records"][0]["target_translation_m"][0] *= -1
    elif change == "anchor":
        candidate["records"][0]["id"] = "another-anchor"
    elif change == "protocol":
        candidate["protocol"] = "frames"
    else:
        candidate["records"][0]["id"] = candidate["records"][1]["id"]
    with pytest.raises(ValueError):
        module.compare(reference, candidate, draws=100)
