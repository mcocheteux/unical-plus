"""Main must see exactly the published anchors used by branch temporal controls."""

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    "window_c2l", Path(__file__).resolve().parents[1] / "experiments/window_c2l.py"
)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


def test_window_anchors_match_causal_context_and_keep_target():
    rows = [
        dict(
            window_id="a",
            window_start_frame=10,
            raw_frame_start_index=110,
            window_length=5,
            gt="target-a",
        ),
        dict(
            window_id="b",
            window_start_frame=15,
            raw_frame_start_index=115,
            window_length=2,
            gt="target-b",
        ),
    ]
    anchors = module.expand_anchors(rows, 3)
    assert [row["frame_index"] for row in anchors] == [12, 13, 14]
    assert [row["raw_frame_index"] for row in anchors] == [112, 113, 114]
    assert {row["sample_id"] for row in anchors} == {"a"}
    assert {row["gt"] for row in anchors} == {"target-a"}
    assert "frame_index" not in rows[0]


def test_invalid_context_rejected():
    with pytest.raises(ValueError, match="positive"):
        module.expand_anchors([], 0)
