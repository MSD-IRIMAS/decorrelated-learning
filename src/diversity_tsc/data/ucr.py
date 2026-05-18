"""UCR Archive data loading and preprocessing.

The UCR Archive root directory can be configured via:

1. The ``ucr_root`` argument passed explicitly,
2. The ``UCR_ARCHIVE_ROOT`` environment variable,
3. A default of ``~/.cache/ucr_archive/UCRArchive_2018``.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import torch
from sklearn.preprocessing import LabelEncoder
from torch.utils.data import DataLoader, TensorDataset

__all__ = [
    "DEFAULT_UCR_ROOT",
    "get_ucr_root",
    "load_ucr_dataset",
    "znormalise",
    "encode_labels",
    "make_dataloader",
]


DEFAULT_UCR_ROOT = Path.home() / ".cache" / "ucr_archive" / "UCRArchive_2018"


def get_ucr_root(ucr_root: Optional[os.PathLike] = None) -> Path:
    """Resolve the UCR Archive root directory.

    Resolution order: explicit argument, ``UCR_ARCHIVE_ROOT`` env var, default.
    """
    if ucr_root is not None:
        return Path(ucr_root).expanduser()
    env = os.environ.get("UCR_ARCHIVE_ROOT")
    if env:
        return Path(env).expanduser()
    return DEFAULT_UCR_ROOT


def load_ucr_dataset(
    dataset_name: str,
    ucr_root: Optional[os.PathLike] = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load a univariate UCR dataset from the standard archive layout.

    Returns
    -------
    xtrain, ytrain, xtest, ytest:
        Numpy arrays with shapes ``(N, T)`` for inputs and ``(N,)`` for labels.

    Raises
    ------
    FileNotFoundError:
        If either the TRAIN or TEST .tsv file is missing.
    """
    root = get_ucr_root(ucr_root) / dataset_name
    train_path = root / f"{dataset_name}_TRAIN.tsv"
    test_path = root / f"{dataset_name}_TEST.tsv"

    if not train_path.is_file() or not test_path.is_file():
        raise FileNotFoundError(
            f"UCR dataset '{dataset_name}' not found under {root}. "
            "Set UCR_ARCHIVE_ROOT or pass --ucr-root."
        )

    train = np.loadtxt(train_path, dtype=np.float64)
    test = np.loadtxt(test_path, dtype=np.float64)

    ytrain, xtrain = train[:, 0], train[:, 1:]
    ytest, xtest = test[:, 0], test[:, 1:]
    return xtrain, ytrain, xtest, ytest


def znormalise(x: np.ndarray) -> np.ndarray:
    """Per-series z-normalisation (zero-mean, unit-variance along the time axis).

    Series with zero standard deviation are returned mean-centred only, which
    avoids dividing by zero.
    """
    mean = x.mean(axis=1, keepdims=True)
    std = x.std(axis=1, keepdims=True)
    std = np.where(std == 0.0, 1.0, std)
    return (x - mean) / std


def encode_labels(y: np.ndarray) -> np.ndarray:
    """Map arbitrary class labels to ``0..C-1`` integer codes."""
    return LabelEncoder().fit_transform(y)


def make_dataloader(
    x: np.ndarray,
    y: np.ndarray,
    batch_size: int = 64,
    shuffle: bool = True,
    seed: int = 42,
    num_workers: int = 0,
) -> DataLoader:
    """Build a deterministic ``DataLoader`` for z-normalised UCR data."""
    x = znormalise(x)
    x = np.expand_dims(x, axis=1)  # (N, 1, T)
    y = encode_labels(y)

    x_t = torch.from_numpy(x).float()
    y_t = torch.from_numpy(y).long()

    generator = torch.Generator().manual_seed(seed)
    return DataLoader(
        TensorDataset(x_t, y_t),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        generator=generator,
        drop_last=False,
    )
