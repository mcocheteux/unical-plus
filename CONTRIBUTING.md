# Contributing

## Development Setup

```bash
git clone https://github.com/mcocheteux/unical-plus.git
cd unical-plus
uv sync --extra dev       # installs package + ruff + pytest
```

Or with pip:

```bash
pip install -e ".[dev]"
```

## Running Tests

```bash
uv run pytest             # run the full suite
uv run pytest -v          # verbose output
uv run pytest tests/test_losses.py   # single file
```

Tests use synthetic tensors and temporary KITTI/parquet fixtures; no real
KITTI download is required. CUDA checks in `tests/test_gpu_training.py` run
on the local GPU when available and skip otherwise:

```bash
CUDA_VISIBLE_DEVICES=0 uv run pytest -v tests/test_gpu_training.py
```

For a real-data training/validation/checkpoint/test smoke run on a local GPU:

```bash
uv run python train.py data_dir=/path/to/kitti_raw experiment=debug \
    data.sequence_length=3 data.batch_size=1 data.num_workers=2 \
    model.temporal.fusion_type=transformer trainer.accelerator=gpu \
    trainer.precision=bf16-mixed trainer.max_epochs=2 \
    +trainer.limit_train_batches=4 +trainer.limit_val_batches=2 \
    +trainer.limit_test_batches=2
```

Evaluate with the same data and model overrides as training. The debug split
uses drives 1 and 5, and shares drive 5 between validation and test; its
metrics establish pipeline functionality, not held-out benchmark accuracy.

## Linting

```bash
uv run ruff check .       # lint
uv run ruff format .      # auto-format
```

Line length is 100 characters (`pyproject.toml`).

## Code Style

- **Type hints** on all public functions and methods.
- **Docstrings** for all public classes and non-trivial methods; include an
  `Args:` block when there are parameters whose meaning is not obvious.
- **No inline comments** that just restate what the code does.  Only add a
  comment when the *why* is non-obvious (a subtle invariant, a paper reference,
  a workaround for a known issue).
- `from __future__ import annotations` at the top of every module.

## Adding Tests

Tests live in `tests/` and use plain `pytest` (no fixtures file).  Follow the
`_fake_batch()` pattern in `tests/test_losses.py` to build synthetic inputs
without touching disk.

## Pull Requests

- Branch from `main`.
- One logical change per PR.
- All tests must pass (`uv run pytest`).
- Ruff must report no new errors (`uv run ruff check .`).
- New public functions need a docstring.
