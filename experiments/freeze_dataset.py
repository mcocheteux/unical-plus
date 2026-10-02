"""Freeze the complete published C2L comparison only after verifying raw coverage."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import tempfile
from pathlib import Path

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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--raw-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    manifest = freeze(args.source, args.raw_root, args.output)
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
