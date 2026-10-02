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
        reference_totals = np.array([sum(a[metric] for a, _ in group) for group in groups.values()])
        contrasts = (totals - 0.1 * reference_totals)[sampled].sum(1) / counts[sampled].sum(1)
        baseline = float(np.mean([r[metric] for r in lhs.values()]))
        delta = float(totals.sum() / counts.sum())
        low, high = np.quantile(means, [0.025, 0.975]).tolist()
        result["metrics"][metric] = {
            "reference": baseline,
            "candidate": baseline + delta,
            "candidate_minus_reference": delta,
            "paired_95_percent_interval": [low, high],
            "noninferiority_margin_10_percent": baseline * 0.1,
            "noninferiority_contrast_95_percent_interval": np.quantile(
                contrasts, [0.025, 0.975]
            ).tolist(),
            "supports_improvement_for_this_seed": high < 0,
            "supports_noninferiority_for_this_seed": bool(np.quantile(contrasts, 0.975) <= 0),
        }
    return result


def compare_replicates(
    references: list[dict], candidates: list[dict], draws: int = 5000, seed: int = 42
) -> dict:
    """Resample paired training seeds and shared test blocks independently.

    Test blocks are shared across selected seeds, preserving the fact that every
    replicate sees the same held-out scenes. Seed replicates receive equal weight.
    At least three distinct paired training seeds are required by our protocol.
    """

    def by_seed(runs: list[dict]) -> dict:
        if any(run.get("training_seed") is None for run in runs):
            raise ValueError("Every replicate must record its training seed")
        indexed = {run["training_seed"]: run for run in runs}
        if len(indexed) != len(runs):
            raise ValueError("Duplicate training seed")
        return indexed

    lhs, rhs = by_seed(references), by_seed(candidates)
    if lhs.keys() != rhs.keys() or len(lhs) < 3:
        raise ValueError("At least three matching training seeds are required")
    seeds = sorted(lhs)
    first = lhs[seeds[0]]
    # Reuse the strict target/anchor/cluster validation for both arms and seeds.
    for training_seed in seeds:
        compare(lhs[training_seed], rhs[training_seed], draws=2)
        compare(first, lhs[training_seed], draws=2)
    groups = {}
    for row in first["records"]:
        groups.setdefault(row["cluster"], []).append(row["id"])
    counts = np.array([len(ids) for ids in groups.values()])
    rng = np.random.default_rng(seed)
    selected_seeds = rng.integers(len(seeds), size=(draws, len(seeds)))
    selected_groups = rng.integers(len(groups), size=(draws, len(groups)))
    denominator = len(seeds) * counts[selected_groups].sum(1)

    def bootstrap(totals: np.ndarray) -> np.ndarray:
        return (
            totals[selected_seeds[:, :, None], selected_groups[:, None, :]].sum((1, 2))
            / denominator
        )

    result = {
        "training_seeds": seeds,
        "samples_per_seed": len(first["records"]),
        "clusters": len(groups),
        "bootstrap_draws": draws,
        "uncertainty_scope": "Paired training seeds and shared held-out temporal blocks.",
        "metrics": {},
    }
    for metric in ["translation_mae_cm", "rotation_degrees"]:
        totals = []
        for runs in [lhs, rhs]:
            seed_totals = []
            for training_seed in seeds:
                records = {r["id"]: r for r in runs[training_seed]["records"]}
                seed_totals.append(
                    [sum(records[k][metric] for k in ids) for ids in groups.values()]
                )
            totals.append(np.array(seed_totals))
        reference_totals, candidate_totals = totals
        differences = candidate_totals - reference_totals
        intervals = np.quantile(bootstrap(differences), [0.025, 0.975]).tolist()
        contrasts = np.quantile(
            bootstrap(candidate_totals - 1.1 * reference_totals), [0.025, 0.975]
        ).tolist()
        baseline = float(reference_totals.sum() / (len(seeds) * counts.sum()))
        candidate = float(candidate_totals.sum() / (len(seeds) * counts.sum()))
        result["metrics"][metric] = {
            "reference": baseline,
            "candidate": candidate,
            "candidate_minus_reference": candidate - baseline,
            "paired_95_percent_interval": intervals,
            "noninferiority_margin_10_percent": 0.1 * baseline,
            "noninferiority_contrast_95_percent_interval": contrasts,
            "supports_improvement": intervals[1] < 0,
            "supports_noninferiority": contrasts[1] <= 0,
        }
    return result


def main() -> None:
    """Save a paired result without substituting per-frame independence for blocks."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, nargs="+", required=True)
    parser.add_argument("--candidate", type=Path, nargs="+", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    references = [json.loads(p.read_text()) for p in args.reference]
    candidates = [json.loads(p.read_text()) for p in args.candidate]
    if len(references) == len(candidates) == 1:
        result = compare(references[0], candidates[0])
    else:
        result = compare_replicates(references, candidates)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
