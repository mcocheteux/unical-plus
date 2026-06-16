# Contributing

## Development Setup

```bash
git clone https://github.com/mcocheteux/unical.git
cd unical
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

Tests use synthetic tensors only — no KITTI data required.

To exercise the full data pipeline end-to-end, create a minimal synthetic
KITTI fixture (see the instructions in `AGENTS.md`), then run:

```bash
python train.py data_dir=<fixture_dir> experiment=debug \
    trainer.max_epochs=1 trainer.accelerator=cpu data.num_workers=0
```

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
