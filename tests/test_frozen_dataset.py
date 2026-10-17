"""Complete comparisons must reject missing assets, split leakage and mutated snapshots."""

import importlib.util
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

spec = importlib.util.spec_from_file_location(
    "freeze_dataset", Path(__file__).resolve().parents[1] / "experiments/freeze_dataset.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


@pytest.fixture
def corpus(tmp_path, monkeypatch):
    source, raw = tmp_path / "source", tmp_path / "raw"
    source.mkdir()
    for split, observations in [
        ("train", [("04", "0001", 0), ("07", "0001", 1)]),
        ("test", [("09", "0002", 0)]),
    ]:
        frames, windows = [], []
        for sequence, drive, frame in observations:
            directory = raw / "2011_09_30" / f"2011_09_30_drive_{drive}_sync"
            directory.mkdir(parents=True, exist_ok=True)
            (directory / ".unical_complete.json").write_text("{}")
            for sensor, suffix in [("image_02", "png"), ("velodyne_points", "bin")]:
                path = directory / sensor / "data" / f"{frame:010d}.{suffix}"
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(b"fixture")
            for stage in range(1, 6):
                row = dict(
                    sequence=sequence,
                    stage=stage,
                    raw_date="2011_09_30",
                    raw_drive=drive,
                    camera_id=2,
                )
                frames.append(dict(row, raw_frame_index=frame))
                windows.append(
                    dict(
                        row,
                        window_id=f"{sequence}-{stage}",
                        raw_frame_start_index=frame,
                        window_length=1,
                    )
                )
        pq.write_table(pa.Table.from_pylist(frames), source / f"{split}.parquet")
        pq.write_table(pa.Table.from_pylist(windows), source / f"windows_{split}.parquet")
    monkeypatch.setattr(
        module.subprocess, "check_output", lambda *args, **kwargs: "fixture-commit\n"
    )
    return source, raw, tmp_path / "frozen"


def test_complete_snapshot_can_be_rechecked_and_rejects_mutation(corpus):
    source, raw, output = corpus
    manifest = module.freeze(source, raw, output)
    assert manifest["unique_frames"] == {"train": 2, "test": 1}
    assert module.freeze(source, raw, output) == manifest
    (output / "train.parquet").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="metadata changed"):
        module.freeze(source, raw, output)


def test_marker_does_not_hide_a_missing_frame(corpus):
    source, raw, output = corpus
    next(raw.rglob("*.png")).unlink()
    with pytest.raises(FileNotFoundError):
        module.freeze(source, raw, output)
    assert not output.exists()


def test_windows_cannot_escape_split(corpus):
    source, raw, output = corpus
    rows = pq.read_table(source / "windows_test.parquet").to_pylist()
    rows[0]["window_length"] = 2
    pq.write_table(pa.Table.from_pylist(rows), source / "windows_test.parquet")
    with pytest.raises(ValueError, match="escapes its split"):
        module.freeze(source, raw, output)


def test_raw_frame_overlap_is_rejected(corpus):
    source, raw, output = corpus
    for name in ["test.parquet", "windows_test.parquet"]:
        rows = pq.read_table(source / name).to_pylist()
        for row in rows:
            row["raw_drive"] = "0001"
        pq.write_table(pa.Table.from_pylist(rows), source / name)
    with pytest.raises(ValueError, match="train/test frames overlap"):
        module.freeze(source, raw, output)


def test_training_view_excludes_whole_sequence_and_keeps_test_bytes(corpus):
    source, raw, frozen = corpus
    drive = raw / "2011_09_30/2011_09_30_drive_0003_sync"
    for relative in [
        ".unical_complete.json",
        "image_02/data/0000000000.png",
        "velodyne_points/data/0000000000.bin",
    ]:
        path = drive / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("fixture")
    for name in ["train.parquet", "windows_train.parquet"]:
        rows = pq.read_table(source / name).to_pylist()
        extra = [
            dict(row, sequence="08", raw_drive="0003") for row in rows if row["sequence"] == "04"
        ]
        if name.startswith("windows_"):
            for row in extra:
                row["window_id"] = f"08-{row['stage']}"
        pq.write_table(pa.Table.from_pylist(rows + extra), source / name)
    module.freeze(source, raw, frozen)
    output = frozen.parent / "without-08"
    manifest = module.derive_training_subset(frozen, output, ["08"])
    assert manifest["unique_frames"] == {"train": 2, "test": 1}
    assert module.derive_training_subset(frozen, output, ["08"]) == manifest
    for name in ["train.parquet", "windows_train.parquet"]:
        assert set(pq.read_table(output / name)["sequence"].to_pylist()) == {"04", "07"}
    for name in ["test.parquet", "windows_test.parquet"]:
        assert (frozen / name).read_bytes() == (output / name).read_bytes()
    (output / "train.parquet").write_bytes(b"mutated")
    with pytest.raises(ValueError, match="subset metadata changed"):
        module.derive_training_subset(frozen, output, ["08"])


def test_training_view_cannot_remove_validation_or_invent_sequence(corpus):
    source, raw, frozen = corpus
    module.freeze(source, raw, frozen)
    for sequence, message in [("07", "Retain sequence 07"), ("08", "must exist")]:
        with pytest.raises(ValueError, match=message):
            module.derive_training_subset(frozen, frozen.parent / "subset", [sequence])
