# Changelog

All notable changes to this project are documented in this file.

The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/).

## [0.2.0] — 2025-05-18

### Added
- Unified `diversity-tsc train` CLI replacing the six monolithic scripts.
- `scripts/run_ensemble.py` orchestrator that chains base training and
  co-training stages automatically.
- Proper `src/` package layout with sub-packages for `models`, `data`,
  `training`, and `utils`.
- Type hints and NumPy-style docstrings across the public API.
- Logging via the standard `logging` module (configurable verbosity).
- 24-test pytest suite covering models, losses, data preprocessing, and the
  training loop.
- GitHub Actions CI workflow (lint + tests on Python 3.9–3.12).
- `pyproject.toml`, `Makefile`, `pre-commit`, and `ruff` configuration.

### Changed
- `LITE` no longer hard-codes `113` channels for its first BatchNorm layer;
  the value is derived from the underlying inception/hybrid blocks.
- The UCR Archive path is now configurable via `--ucr-root` or the
  `UCR_ARCHIVE_ROOT` environment variable instead of being hard-coded.
- The diversity penalty is implemented as a single reusable function
  (`feature_diversity_penalty`) rather than being duplicated per script.

### Fixed
- Off-by-one bug in the original `cotrain_5.py` where the fourth penalty term
  used the third penalty's mask for its non-zero count.
- `base.py` previously placed the seed loop outside `if __name__ == "__main__"`,
  causing every import to run training.

### Removed
- `base.py`, `cotrain.py`, `cotrain_2.py`, `cotrain_3.py`, `cotrain_4.py`,
  `cotrain_5.py` (superseded by the unified CLI).

## [0.1.0] — 2024-12-01

### Added
- Initial release accompanying the IJCNN 2025 submission.
