"""Tests for UCR data loading and preprocessing."""


import numpy as np
import pytest

from diversity_tsc.data import (
    DEFAULT_UCR_ROOT,
    encode_labels,
    get_ucr_root,
    load_ucr_dataset,
    make_dataloader,
    znormalise,
)


def test_get_ucr_root_default_when_no_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("UCR_ARCHIVE_ROOT", raising=False)
    assert get_ucr_root() == DEFAULT_UCR_ROOT


def test_get_ucr_root_explicit_argument_wins(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("UCR_ARCHIVE_ROOT", "/tmp/wrong")
    result = get_ucr_root(tmp_path)
    assert result == tmp_path


def test_get_ucr_root_uses_env(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setenv("UCR_ARCHIVE_ROOT", str(tmp_path))
    assert get_ucr_root() == tmp_path


def test_znormalise_zero_mean_unit_std() -> None:
    x = np.random.randn(10, 100) * 5 + 3
    z = znormalise(x)
    assert np.allclose(z.mean(axis=1), 0, atol=1e-6)
    assert np.allclose(z.std(axis=1), 1, atol=1e-6)


def test_znormalise_handles_constant_series() -> None:
    x = np.array([[1.0, 1.0, 1.0], [2.0, 4.0, 6.0]])
    z = znormalise(x)
    # First row is constant -> should not produce NaNs/Infs
    assert np.all(np.isfinite(z))
    # Second row should be properly normalised
    assert abs(z[1].mean()) < 1e-6


def test_encode_labels_maps_to_consecutive_integers() -> None:
    y = np.array([3, 1, 1, 7, 3])
    e = encode_labels(y)
    assert set(e.tolist()) == {0, 1, 2}


def test_load_ucr_dataset_missing_raises(tmp_path) -> None:
    with pytest.raises(FileNotFoundError):
        load_ucr_dataset("DoesNotExist", ucr_root=tmp_path)


def test_make_dataloader_shapes_and_types() -> None:
    x = np.random.randn(32, 50)
    y = np.random.randint(0, 3, size=32)
    loader = make_dataloader(x, y, batch_size=8)
    batch_x, batch_y = next(iter(loader))
    assert batch_x.shape == (8, 1, 50)
    assert batch_y.dtype.is_floating_point is False  # long
    assert batch_x.dtype.is_floating_point  # float
