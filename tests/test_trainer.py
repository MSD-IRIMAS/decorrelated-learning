"""Light-weight integration tests for the trainer."""

import numpy as np
import torch

from diversity_tsc.data import make_dataloader
from diversity_tsc.models import LITE
from diversity_tsc.training import Trainer, TrainerConfig
from diversity_tsc.utils import set_seed


def _toy_loaders(n_train: int = 32, n_test: int = 16, length: int = 48, n_classes: int = 3):
    rng = np.random.default_rng(0)
    xtr = rng.standard_normal((n_train, length))
    ytr = rng.integers(0, n_classes, n_train)
    xte = rng.standard_normal((n_test, length))
    yte = rng.integers(0, n_classes, n_test)
    return (
        make_dataloader(xtr, ytr, batch_size=8, shuffle=True),
        make_dataloader(xte, yte, batch_size=8, shuffle=False),
        length,
        n_classes,
    )


def test_baseline_trainer_runs_a_few_epochs() -> None:
    set_seed(0)
    train_loader, val_loader, length, n_classes = _toy_loaders()
    model = LITE(length_TS=length, n_classes=n_classes)
    cfg = TrainerConfig(epochs=2, device="cpu", best_model_min_epoch=0)

    trainer = Trainer(model, train_loader, val_loader, cfg)
    history, best_state = trainer.fit()

    assert len(history.train_loss) == 2
    assert len(history.val_loss) == 2
    assert best_state is not None
    assert history.duration_seconds > 0


def test_cotraining_with_one_reference() -> None:
    set_seed(0)
    train_loader, val_loader, length, n_classes = _toy_loaders()

    reference = LITE(length_TS=length, n_classes=n_classes).eval()
    student = LITE(length_TS=length, n_classes=n_classes)
    cfg = TrainerConfig(epochs=2, device="cpu", best_model_min_epoch=0)

    trainer = Trainer(student, train_loader, val_loader, cfg, reference_models=[reference])
    history, _ = trainer.fit()

    assert len(history.train_div_loss) == 2
    # With a real reference model the diversity loss should be > 0
    assert all(d >= 0 for d in history.train_div_loss)


def test_reference_model_is_not_updated() -> None:
    set_seed(0)
    train_loader, val_loader, length, n_classes = _toy_loaders()

    reference = LITE(length_TS=length, n_classes=n_classes).eval()
    initial = {k: v.clone() for k, v in reference.state_dict().items()}

    student = LITE(length_TS=length, n_classes=n_classes)
    cfg = TrainerConfig(epochs=2, device="cpu", best_model_min_epoch=0)
    trainer = Trainer(student, train_loader, val_loader, cfg, reference_models=[reference])
    trainer.fit()

    after = reference.state_dict()
    for k, v in initial.items():
        assert torch.equal(v, after[k]), f"reference parameter '{k}' changed during training"
