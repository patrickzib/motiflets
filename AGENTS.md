# Repository Guidelines

## Project Structure & Module Organization
- `motiflets/` holds the Python package; `motiflets.py` drives the algorithms while `distances.py`, `competitors.py`, and `plotting.py` supply helpers. Keep new core code here.
- `tests/` mirrors the package layout and includes fixture assets under subfolders such as `csv`, `images_paper`, and `results`. Prefer lightweight fixtures when adding cases.
- `datasets/`, `csvs/`, `images/`, and `notebooks/` host paper resources. Treat them as read-only unless a manuscript update explicitly requires new material.
- Build artefacts live in `build/` and `dist/`. Remove local scratch outputs before committing.

## Build, Test, and Development Commands
- `python -m pip install -e .` installs the package in editable mode for local development.
- `python -m build` produces source and wheel distributions in `dist/`; run this before publishing artefacts.
- `pytest -q` runs the full suite. For a quick pass, use `pytest -q -k "not scalability"` to skip the long-running benchmark.
- `python -m pytest tests/test_scalability.py -s` reproduces the timing study and requires data from `datasets/experiments/`.

## Coding Style & Naming Conventions
- Target Python 3.7+ and follow PEP 8: 4-space indentation, snake_case for functions and variables, PascalCase for new classes.
- Match the existing NumPy-style docstrings (see `motiflets/motiflets.py`) and keep narrative comments concise and purposeful.
- Prefer vectorised NumPy/Pandas operations; fall back to explicit loops only with a short rationale.
- Maintain compatibility with both NumPy arrays and pandas Series/DataFrames, as the public API expects either.

## Testing Guidelines
- Extend `pytest` coverage for every bug fix or feature; co-locate new tests beside related modules (e.g., `tests/test_input_dimensions.py` patterns for interface checks).
- Use descriptive test names like `test_<condition>_<expected_behavior>` and assert on both values and shapes.
- Place bulky fixtures in `tests/` subdirectories and document provenance if new experimental data is needed.
- Long-running scalability tests should stay skipped during routine CI; gate them behind explicit `-k` selections.

## Commit & Pull Request Guidelines
- Follow the existing history: short, imperative present-tense subjects (`add sliding dot benchmark`) under 72 characters.
- Group related changes per commit and include relevant test updates.
- Pull requests should list the motivation, key changes, test evidence (`pytest -q`, targeted runs), and reference paper sections or issues when applicable.
