"""Tests for the feature-diversity penalty."""

import pytest
import torch

from diversity_tsc.training.losses import (
    feature_diversity_penalty,
    pairwise_diversity_penalty,
)


def test_pairwise_penalty_is_non_negative() -> None:
    a = torch.randn(8, 16, 32)
    b = torch.randn(8, 16, 32)
    penalty = pairwise_diversity_penalty(a, b)
    assert penalty.item() >= 0


def test_pairwise_penalty_shape_mismatch_raises() -> None:
    a = torch.randn(8, 16, 32)
    b = torch.randn(8, 8, 32)
    with pytest.raises(ValueError):
        pairwise_diversity_penalty(a, b)


def test_feature_diversity_with_no_references_returns_zero() -> None:
    a = torch.randn(2, 4, 8)
    penalty = feature_diversity_penalty(a, [])
    assert penalty.item() == 0.0


def test_feature_diversity_averages_over_references() -> None:
    torch.manual_seed(0)
    a = torch.randn(4, 8, 16, requires_grad=True)
    refs = [torch.randn(4, 8, 16) for _ in range(3)]
    pair = torch.stack([pairwise_diversity_penalty(a, r) for r in refs]).mean()
    multi = feature_diversity_penalty(a, refs)
    assert torch.allclose(pair, multi, atol=1e-6)


def test_diversity_penalty_is_differentiable() -> None:
    a = torch.randn(2, 4, 8, requires_grad=True)
    refs = [torch.randn(2, 4, 8) for _ in range(2)]
    penalty = feature_diversity_penalty(a, refs)
    penalty.backward()
    assert a.grad is not None
    assert a.grad.abs().sum() > 0


def test_references_are_detached_internally() -> None:
    """Reference features should not accumulate gradient even if they require it."""
    a = torch.randn(2, 4, 8, requires_grad=True)
    ref = torch.randn(2, 4, 8, requires_grad=True)
    penalty = feature_diversity_penalty(a, [ref])
    penalty.backward()
    assert ref.grad is None
