# Repository Guidelines

## Project Structure & Module Organization
- `motiflets/` is the core Python package; `motiflets.py` orchestrates motif discovery while `distances.py`, `competitors.py`, and backend modules provide supporting algorithms.
- `tests/` holds scenario-driven pytest suites (e.g., `test_pyattimo_*`, `test_scalability.py`) and their companion assets; keep generated artifacts inside the existing subfolders.
- `notebooks/` contains paper-aligned demos; treat notebooks as read-only unless syncing updates from published experiments.
- `datasets/`, `csvs/`, `images/`, and `jars/` store published data, plots, and external binaries—do not modify in-place; add new assets under a dated subdirectory.
- `build/`, `dist/`, and `motiflets.egg-info/` are build outputs; regenerate them via the build commands instead of editing by hand.

## Build, Test, and Development Commands
- `python -m venv .venv && source .venv/bin/activate` creates an isolated development environment.
- `python -m pip install -e .` installs the package in editable mode with required runtime dependencies.
- `python -m build` produces the sdist and wheel inside `dist/` for release validation.
- `pytest -q` runs the full suite; scope runs with `pytest -k pyattimo_ecg` or `pytest tests/test_scalability.py --maxfail=1` for focused checks.

## Coding Style & Naming Conventions
- Follow PEP 8 with four-space indentation and descriptive, lowercase identifiers; mirror naming used in `motiflets/distances.py` and `motiflets/motiflets.py`.
- Keep public API signatures stable and document them with concise NumPy-style docstrings; prefer keyword arguments for optional parameters.
- Vectorized NumPy/Numba sections must include brief inline comments that explain non-obvious indexing or memory optimizations.
- Reuse plotting conventions from `motiflets/plotting.py`; keep figure names snake_case.

## Testing Guidelines
- Pytest is configured via `[tool.pytest.ini_options]` in `pyproject.toml`; rely on CLI logging to trace long runs.
- Place new tests beside related modules (e.g., `tests/test_stitching.py`) and adopt the `test_<feature>.py` pattern.
- Large fixtures belong in the structured subfolders already present under `tests/`; prefer CSV or NumPy formats and include README notes when adding datasets.
- When benchmarking, call helpers in `tests/utils.py` to manage backend parameters and enforce deterministic seeds for reproducible motif sets.

## Commit & Pull Request Guidelines
- Write imperative, scoped commit subjects such as `motiflets: tighten elbow selection`; avoid vague messages like "updates".
- Group related changes per commit, reference issue IDs when relevant, and note any dataset or notebook artifacts touched.
- Pull requests should list motivation, highlight algorithmic or data changes, and document test evidence (e.g., `pytest -q`).
- Include screenshots or regenerated figures when adjusting visualization routines so reviewers can confirm graphical diffs.
