"""Paired clustered uncertainty for models evaluated on identical held-out targets."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np


def compare(reference: dict, candidate: dict, draws: int = 5000, seed: int = 42) -> dict:
    """Bootstrap paired mean error differences by temporal block, preserving sample pairs.

    Multiple training seeds must be assessed separately; this interval measures
    test-block variability for one seed and does not claim training uncertainty.
    """
    if reference["protocol"] != candidate["protocol"] or reference["stage"] != candidate["stage"]:
        raise ValueError("Evaluation protocol/stage must match")
    lhs = {row["id"]: row for row in reference["records"]}
    rhs = {row["id"]: row for row in candidate["records"]}
    if len(lhs) != len(reference["records"]) or len(rhs) != len(candidate["records"]):
        raise ValueError("Duplicate sample IDs")
    if lhs.keys() != rhs.keys():
        raise ValueError("Different evaluation anchors")
    groups = {}
    for key, row in lhs.items():
        other = rhs[key]
        if row["cluster"] != other["cluster"]:
            raise ValueError("Mismatched bootstrap cluster")
        for target in ["target_translation_m", "target_rotation_matrix"]:
            if not np.allclose(row[target], other[target], rtol=1e-6, atol=1e-7):
                raise ValueError("Evaluation targets differ")
        for baseline in ["initial_translation_mae_cm", "initial_rotation_degrees"]:
            if not np.isclose(row[baseline], other[baseline], rtol=1e-6, atol=1e-6):
                raise ValueError("Evaluation targets differ")
        groups.setdefault(row["cluster"], []).append((row, other))
    if len(groups) < 2:
        raise ValueError("At least two independent temporal blocks are required")
    rng = np.random.default_rng(seed)
    sampled = rng.integers(0, len(groups), size=(draws, len(groups)))
    counts = np.array([len(g) for g in groups.values()])
    result = {
        "samples": len(lhs),
        "clusters": len(groups),
        "bootstrap_draws": draws,
        "uncertainty_scope": "Test blocks for this seed; repeat across independent training seeds.",
        "metrics": {},
    }
    for metric in ["translation_mae_cm", "rotation_degrees"]:
        totals = np.array(
            [sum(b[metric] - a[metric] for a, b in group) for group in groups.values()]
        )
        means = totals[sampled].sum(1) / counts[sampled].sum(1)
        baseline = float(np.mean([r[metric] for r in lhs.values()]))
        delta = float(totals.sum() / counts.sum())
        low, high = np.quantile(means, [0.025, 0.975]).tolist()
        result["metrics"][metric] = {
            "reference": baseline,
            "candidate": baseline + delta,
            "candidate_minus_reference": delta,
            "paired_95_percent_interval": [low, high],
            "noninferiority_margin_10_percent": baseline * 0.1,
            "supports_improvement_for_this_seed": high < 0,
            "supports_noninferiority_for_this_seed": high <= baseline * 0.1,
        }
    return result


def main() -> None:
    """Save a paired result without substituting per-frame independence for blocks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    result = compare(json.loads(args.reference.read_text()), json.loads(args.candidate.read_text()))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
