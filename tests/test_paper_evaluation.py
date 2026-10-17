"""Paper comparisons must reject C2L/raw-drive leakage and changed metadata."""

import hashlib
import json
import sys
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "experiments"))
from evaluate_comparison import guard_paper_alpha_training  # noqa: E402


def _manifest(directory, drive):
    path = directory / "train.parquet"
    pq.write_table(pa.Table.from_pylist([{"raw_date": "2011_09_30", "raw_drive": drive}]), path)
    return {
        "arguments": {"protocol": "frames", "data_dir": str(directory)},
        "dataset_hashes": {"train.parquet": hashlib.sha256(path.read_bytes()).hexdigest()},
    }


def test_c2l_drive_28_exposure_is_rejected_by_raw_identity(tmp_path):
    with pytest.raises(ValueError, match="exclude it before training"):
        guard_paper_alpha_training(_manifest(tmp_path, "0028"))


def test_disjoint_training_metadata_is_accepted_but_mutation_rejected(tmp_path):
    manifest = _manifest(tmp_path, "0027")
    guard_paper_alpha_training(manifest)
    _manifest(tmp_path, "0028")
    with pytest.raises(ValueError, match="metadata changed"):
        guard_paper_alpha_training(manifest)


def test_usual_split_checks_training_and_validation_exposure():
    manifest = {
        "arguments": {"protocol": "usual"},
        "configuration": {
            "experiment": {
                "splits": {
                    "train": [["2011_09_26", [1, 2]]],
                    "val": [["2011_09_26", [5, 70]]],
                }
            }
        },
    }
    guard_paper_alpha_training(manifest)
    for split in ["train", "val"]:
        leaked = json.loads(json.dumps(manifest))
        leaked["configuration"]["experiment"]["splits"][split] += [["2011_09_30", [28]]]
        with pytest.raises(ValueError, match="training/validation"):
            guard_paper_alpha_training(leaked)


def test_missing_historical_training_hash_is_not_inferred(tmp_path):
    manifest = _manifest(tmp_path, "0027")
    del manifest["dataset_hashes"]
    with pytest.raises(ValueError, match="requires recorded training metadata and its hash"):
        guard_paper_alpha_training(manifest)
