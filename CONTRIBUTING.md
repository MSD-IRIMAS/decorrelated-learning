# Contributing

Thanks for your interest in improving `diversity-tsc`. This document explains
the workflow we follow.

## Development setup

```bash
git clone https://github.com/javidan-abdullayev/diversity-tsc.git
cd diversity-tsc
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pre-commit install
```

## Workflow

1. **Fork** the repository and create a topic branch:
   ```bash
   git checkout -b feat/short-description
   ```
2. Make your changes. Keep commits focused and descriptive.
3. Make sure the test suite, linter, and type-checker pass locally:
   ```bash
   make lint
   make test
   make typecheck
   ```
4. Push and open a pull request against `main` describing the motivation and
   the changes. Link to any related issues.

## Coding conventions

- Follow `ruff` and `ruff format`. Both run via `pre-commit` and CI.
- Use type hints on all new public functions and classes.
- Write a docstring (NumPy style) for every public symbol.
- Keep lines ≤ 100 characters.
- Prefer pure functions for helpers; reserve classes for stateful components.

## Adding tests

Place tests under `tests/`, mirroring the module path. Each new feature should
ship with at least one test. Use `pytest` fixtures over duplicated setup code.

## Reporting bugs

Open an issue with:

- a minimal reproduction (ideally a `pytest` snippet),
- the full traceback,
- your Python and PyTorch versions (`python -V && pip show torch`).

## Releases

Maintainers bump `__version__` in `src/diversity_tsc/_version.py`, tag the
commit (`vX.Y.Z`), and let CI build the wheel.
