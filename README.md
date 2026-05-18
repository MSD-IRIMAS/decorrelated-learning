# Diversity-Driven Ensemble Learning for Time Series Classification

[![CI](https://github.com/javidan-abdullayev/diversity-tsc/actions/workflows/ci.yml/badge.svg)](https://github.com/javidan-abdullayev/diversity-tsc/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-ee4c2c.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

A PyTorch implementation of a diversity-driven ensemble learning framework for
**time series classification (TSC)**. The framework augments the
[LITE](https://arxiv.org/abs/2409.02869) architecture with a feature-decorrelation
penalty that pushes ensemble members to learn complementary representations,
yielding state-of-the-art accuracy on the UCR Archive with fewer ensemble
members than traditional bagging-style ensembles.

> 📝 **Paper:** [Enhancing Time Series Classification with Diversity-Driven Neural Network Ensembles](./docs/IJCNN_2025_Ensemble_Learning_for_TSC.pdf) — IJCNN 2025.

---

## Table of Contents

- [Highlights](#highlights)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Reproducing the Paper](#reproducing-the-paper)
- [Repository Layout](#repository-layout)
- [How It Works](#how-it-works)
- [Programmatic Usage](#programmatic-usage)
- [Development](#development)
- [Citation](#citation)
- [License](#license)

---

## Highlights

- **Single CLI** (`diversity-tsc train`) replaces six monolithic scripts.
- **Modular package** with clean separation between models, data, and training.
- **Configurable ensemble size** (1 to N) with automatic reference-checkpoint wiring.
- **Reproducible**: deterministic seeding, pinned hyperparameters, and a CI suite.
- **Tested**: 24 unit and integration tests covering models, losses, data, and training.
- **Typed**: PEP-604/`typing` annotations throughout the public API.

## Installation

### From source

```bash
git clone https://github.com/javidan-abdullayev/diversity-tsc.git
cd diversity-tsc
pip install -e .
```

For development:

```bash
pip install -e ".[dev]"
pre-commit install
```

### Dataset setup

Download the [UCR Archive 2018](https://www.cs.ucr.edu/~eamonn/time_series_data_2018/)
and point the package at it via any of the following:

```bash
# Option 1: environment variable
export UCR_ARCHIVE_ROOT=/path/to/UCRArchive_2018

# Option 2: per-command flag
diversity-tsc train --ucr-root /path/to/UCRArchive_2018 ...

# Option 3: default location (~/.cache/ucr_archive/UCRArchive_2018)
```

## Quick Start

Train a single baseline LITE model on one dataset:

```bash
diversity-tsc train \
    --datasets ECGFiveDays \
    --ensemble-size 1 \
    --seed 42 \
    --output-directory runs/
```

Train a co-trained ensemble member that decorrelates against two previously
trained checkpoints:

```bash
diversity-tsc train \
    --datasets ECGFiveDays \
    --ensemble-size 3 \
    --seed 8976 \
    --reference-checkpoints \
        runs/base_seed_37/ECGFiveDays/best_model.pt \
        runs/co2_seed_5639/ECGFiveDays/best_model.pt
```

Each run produces, under `runs/<run_name>/<dataset>/`:

- `best_model.pt` / `last_model.pt` — checkpoints
- `history.csv` — per-epoch losses, accuracies, learning rates
- `metrics.csv` — final test accuracy, runtime, best epoch
- `losses.png` / `accuracies.png` — training curves

## Reproducing the Paper

A helper script chains base training and `ensemble_size - 1` co-training stages
automatically, wiring reference checkpoints from the previous step:

```bash
python scripts/run_ensemble.py \
    --datasets ECGFiveDays Coffee \
    --ensemble-size 5 \
    --seeds 37 5639 8976 8859 9876 \
    --output-directory runs/
```

To reproduce the full evaluation on all 128 UCR datasets:

```bash
python scripts/run_ensemble.py \
    --ensemble-size 5 \
    --seeds 37 5639 8976 8859 9876 \
    --output-directory runs/
```

The hyperparameters from the paper are the defaults: 1500 epochs, Adam with
`lr=1e-3`, `ReduceLROnPlateau` (factor 0.5, patience 50, min 1e-4), batch size
64, loss weights `ce=0.8` and `diversity=0.2`, and a diversity penalty scale of
40.

## Repository Layout

```
diversity-tsc/
├── src/diversity_tsc/
│   ├── cli.py                 # diversity-tsc CLI entry point
│   ├── models/
│   │   └── lite.py            # LITE, Inception, Hybrid, FCN blocks
│   ├── data/
│   │   └── ucr.py             # UCR loading + preprocessing
│   ├── training/
│   │   ├── losses.py          # feature-diversity penalty
│   │   └── trainer.py         # unified training loop
│   └── utils/                 # seeding, logging, plotting, dataset names
├── scripts/
│   └── run_ensemble.py        # multi-stage ensemble orchestrator
├── tests/                     # 24 unit + integration tests
├── configs/                   # example YAML configs (optional)
├── docs/                      # paper PDF and design notes
├── pyproject.toml             # build + tooling config
├── requirements.txt
├── Makefile
└── .github/workflows/ci.yml   # lint + test on 3.9–3.12
```

## How It Works

Each ensemble member is a [LITE](https://arxiv.org/abs/2409.02869) classifier —
a lightweight Inception-style network combining standard convolutions with
hand-crafted "hybrid" filters that detect increases, decreases, and peaks at
multiple scales.

After training a baseline model with standard cross-entropy, each subsequent
ensemble member is co-trained against the frozen checkpoints of all previous
members. Co-training minimises a weighted sum of two losses:

```
ℒ = ω_ce · CE(y, ŷ)  +  ω_div · 1/(N-1) Σ_i  Penalty(f_student, f_reference_i)
```

The diversity penalty is the mean off-diagonal Frobenius norm of the dot
product between the student's last feature map and each reference's last
feature map. Forcing this term toward zero pushes the student to span feature
directions that the reference models do not.

At inference, predictions are averaged across all ensemble members.

## Programmatic Usage

```python
import torch
from diversity_tsc.data import load_ucr_dataset, make_dataloader
from diversity_tsc.models import LITE
from diversity_tsc.training import Trainer, TrainerConfig
from diversity_tsc.utils import set_seed

set_seed(42)

xtrain, ytrain, xtest, ytest = load_ucr_dataset("ECGFiveDays")
train_loader = make_dataloader(xtrain, ytrain, batch_size=64, shuffle=True)
val_loader = make_dataloader(xtest, ytest, batch_size=64, shuffle=False)

model = LITE(length_TS=xtrain.shape[1], n_classes=len(set(ytrain)))
cfg = TrainerConfig(epochs=1500)
trainer = Trainer(model, train_loader, val_loader, cfg)

history, best_state = trainer.fit(output_directory="runs/my_run")
val_loss, val_acc = trainer.evaluate()
print(f"Test accuracy: {val_acc:.2f}%")
```

## Development

Common tasks are exposed through the `Makefile`:

```bash
make install-dev    # install with dev deps + pre-commit hooks
make lint           # ruff check
make format         # ruff fix + format
make typecheck      # mypy
make test           # pytest
make test-cov       # pytest + coverage report
make smoke-test     # tiny end-to-end training run
```

Pull requests should pass `make lint && make test` locally; CI runs the same
checks against Python 3.9 / 3.10 / 3.11 / 3.12.

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{abdullayev2025diversity,
  title     = {Enhancing Time Series Classification with Diversity-Driven Neural Network Ensembles},
  author    = {Abdullayev, Javidan and Devanne, Maxime and Meyer, Cyril and
               Ismail-Fawaz, Ali and Weber, Jonathan and Forestier, Germain},
  booktitle = {Proceedings of the International Joint Conference on Neural Networks (IJCNN)},
  year      = {2025}
}
```

## License

Released under the [MIT License](LICENSE).

## Acknowledgments

This work was conducted at Université de Haute-Alsace and supported by the
IRIMAS laboratory. The LITE architecture is adapted from
[MSD-Mix-LITE/LITE](https://github.com/MSD-Mvpcom/LITE) by Ismail-Fawaz et al.
