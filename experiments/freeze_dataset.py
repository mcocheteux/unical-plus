"""Freeze the complete published C2L comparison only after verifying raw coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def freeze(source: Path, raw_root: Path, output: Path) -> dict:
    """Check all frames/windows, reject split leakage, then atomically freeze metadata."""
    names = ["train.parquet", "test.parquet", "windows_train.parquet", "windows_test.parquet"]
    frames = {}
    inventory = {}
    counts = {}
    for split in ["train", "test"]:
        rows = pq.read_table(source / f"{split}.parquet").to_pylist()
        if {row["stage"] for row in rows} != {1, 2, 3, 4, 5}:
            raise ValueError("Complete corpus requires all five stages")
        if split == "train" and (
            "07" not in {row["sequence"] for row in rows}
            or len({row["sequence"] for row in rows}) < 2
        ):
            raise ValueError("Missing sequence 07 validation holdout or training sequences")
        counts[split] = len(rows)
        frames[split] = set()
        for row in rows:
            drive = raw_root / row["raw_date"] / f"{row['raw_date']}_drive_{row['raw_drive']}_sync"
            marker = drive / ".unical_complete.json"
            if not marker.is_file():
                raise FileNotFoundError(f"Incomplete published drive: {drive}")
            inventory[drive] = marker
            key = (drive, int(row["camera_id"]), int(row["raw_frame_index"]))
            frames[split].add(key)
        for drive, camera, index in frames[split]:
            for relative in [
                f"image_0{camera}/data/{index:010d}.png",
                f"velodyne_points/data/{index:010d}.bin",
            ]:
                if not (drive / relative).is_file():
                    raise FileNotFoundError(drive / relative)
        windows = pq.read_table(source / f"windows_{split}.parquet").to_pylist()
        if {row["stage"] for row in windows} != {1, 2, 3, 4, 5}:
            raise ValueError("Complete window corpus requires all five stages")
        counts[f"windows_{split}"] = len(windows)
        for row in windows:
            drive = raw_root / row["raw_date"] / f"{row['raw_date']}_drive_{row['raw_drive']}_sync"
            start = int(row["raw_frame_start_index"])
            for index in range(start, start + int(row["window_length"])):
                if (drive, int(row["camera_id"]), index) not in frames[split]:
                    raise ValueError(f"Published window escapes its split: {row['window_id']}")
    if frames["train"] & frames["test"]:
        raise ValueError("Raw train/test frames overlap")
    if min(counts.values()) == 0:
        raise ValueError("Empty published split")
    hashes = {name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names}
    manifest = {
        "source": str(source.resolve()),
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=source, text=True
        ).strip(),
        "source_hashes": hashes,
        "raw_root": str(raw_root.resolve()),
        "raw_inventory_hashes": {
            str(drive): hashlib.sha256(marker.read_bytes()).hexdigest()
            for drive, marker in sorted(inventory.items())
        },
        "rows": counts,
        "unique_frames": {split: len(values) for split, values in frames.items()},
        "validation_sequence": "07",
        "scope": "Complete published frame and window corpus, all five stages; no raw train/test overlap.",
    }
    if output.exists():
        if json.loads((output / "selection.json").read_text()) != manifest:
            raise FileExistsError(f"Existing snapshot differs: {output}")
        if any(
            hashlib.sha256((output / name).read_bytes()).hexdigest() != hashes[name]
            for name in names
        ):
            raise ValueError("Frozen snapshot metadata changed")
        return manifest
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / "snapshot"
        candidate.mkdir()
        for name in names:
            shutil.copyfile(source / name, candidate / name)
        (candidate / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
        candidate.rename(output)
    return manifest


def derive_training_subset(source: Path, output: Path, excluded: list[str]) -> dict:
    """Derive an immutable training view; retain held-out metadata byte-for-byte."""
    parent_file = source / "selection.json"
    parent = json.loads(parent_file.read_text())
    names = ["train.parquet", "test.parquet", "windows_train.parquet", "windows_test.parquet"]
    expected = parent.get("metadata_hashes", parent["source_hashes"])
    source_hashes = {
        name: hashlib.sha256((source / name).read_bytes()).hexdigest() for name in names
    }
    if source_hashes != expected:
        raise ValueError("Parent snapshot metadata changed")
    excluded = sorted(set(excluded))
    table = pq.read_table(source / "train.parquet")
    sequences = set(table["sequence"].to_pylist())
    if not excluded or not set(excluded) <= sequences:
        raise ValueError("Excluded sequences must exist in the parent training split")
    remaining = sequences - set(excluded)
    if "07" not in remaining or len(remaining) < 2:
        raise ValueError("Retain sequence 07 validation and at least one training sequence")
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output.parent) as temporary:
        candidate = Path(temporary) / "snapshot"
        candidate.mkdir()
        counts = {}
        unique = {}
        for name in names:
            split_table = pq.read_table(source / name)
            if name in ["train.parquet", "windows_train.parquet"]:
                keep = pc.invert(pc.is_in(split_table["sequence"], value_set=pa.array(excluded)))
                split_table = split_table.filter(keep)
                pq.write_table(split_table, candidate / name)
            else:
                shutil.copyfile(source / name, candidate / name)
            counts[name.removesuffix(".parquet")] = len(split_table)
            if not name.startswith("windows_"):
                rows = split_table.select(
                    ["raw_date", "raw_drive", "camera_id", "raw_frame_index"]
                ).to_pylist()
                unique[name.removesuffix(".parquet")] = len({tuple(row.values()) for row in rows})
        hashes = {
            name: hashlib.sha256((candidate / name).read_bytes()).hexdigest() for name in names
        }
        manifest = {
            "source": str(source.resolve()),
            "source_commit": parent["source_commit"],
            "source_hashes": source_hashes,
            "parent_manifest_sha256": hashlib.sha256(parent_file.read_bytes()).hexdigest(),
            "metadata_hashes": hashes,
            "excluded_training_sequences": excluded,
            "raw_root": parent["raw_root"],
            "raw_inventory_hashes": parent["raw_inventory_hashes"],
            "rows": counts,
            "unique_frames": unique,
            "validation_sequence": "07",
            "scope": "Whole-sequence training exclusion; unchanged targets, causal windows and held-out metadata. Not an exact paper training split.",
        }
        if output.exists():
            if json.loads((output / "selection.json").read_text()) != manifest:
                raise FileExistsError(f"Existing subset differs: {output}")
            if any(
                hashlib.sha256((output / name).read_bytes()).hexdigest() != hashes[name]
                for name in names
            ):
                raise ValueError("Frozen subset metadata changed")
            return manifest
        (candidate / "selection.json").write_text(json.dumps(manifest, indent=2) + "\n")
        candidate.rename(output)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--exclude-training-sequences", nargs="+")
    args = parser.parse_args()
    if args.exclude_training_sequences:
        parent = json.loads((args.source / "selection.json").read_text())
        if Path(parent["raw_root"]).resolve() != args.raw_root.resolve():
            parser.error("Subset raw root must match its parent snapshot")
    manifest = (
        derive_training_subset(args.source, args.output, args.exclude_training_sequences)
        if args.exclude_training_sequences
        else freeze(args.source, args.raw_root, args.output)
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
