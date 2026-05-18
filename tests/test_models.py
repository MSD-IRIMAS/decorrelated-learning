"""Tests for the LITE model architecture."""

import pytest
import torch

from diversity_tsc.models import LITE


@pytest.mark.parametrize("length_TS,n_classes", [(96, 5), (251, 2), (1000, 10)])
def test_lite_forward_shapes(length_TS: int, n_classes: int) -> None:
    model = LITE(length_TS=length_TS, n_classes=n_classes)
    x = torch.randn(4, 1, length_TS)
    logits, gap = model(x)
    assert logits.shape == (4, n_classes)
    assert gap.shape == (4, model.n_filters)


def test_lite_extract_features_returns_three_maps() -> None:
    model = LITE(length_TS=128, n_classes=3)
    x = torch.randn(2, 1, 128)
    logits, features = model.extract_features(x)
    assert logits.shape == (2, 3)
    assert len(features) == 3
    # The last feature map is what the diversity penalty uses
    assert features[-1].dim() == 3  # (B, C, T)
    assert features[-1].shape[0] == 2


def test_lite_backward_pass() -> None:
    model = LITE(length_TS=64, n_classes=2)
    x = torch.randn(2, 1, 64)
    logits, _ = model(x)
    loss = logits.sum()
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.requires_grad]
    assert any(g is not None and g.abs().sum() > 0 for g in grads)


def test_lite_without_hybrid_layer() -> None:
    """Sanity check: model still works with hand-crafted filters disabled."""
    model = LITE(length_TS=128, n_classes=4, use_custom_filters=False)
    x = torch.randn(2, 1, 128)
    logits, _ = model(x)
    assert logits.shape == (2, 4)


def test_lite_deterministic_with_seed() -> None:
    torch.manual_seed(0)
    model_a = LITE(length_TS=64, n_classes=2)
    torch.manual_seed(0)
    model_b = LITE(length_TS=64, n_classes=2)
    for pa, pb in zip(model_a.parameters(), model_b.parameters()):
        assert torch.equal(pa, pb)
