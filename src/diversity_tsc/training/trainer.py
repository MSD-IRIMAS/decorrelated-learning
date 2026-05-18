"""Unified training loop for base and co-trained LITE models."""

from __future__ import annotations

import copy
import logging
import time
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, Tuple

import pandas as pd
import torch
from torch import nn
from torch.utils.data import DataLoader

from diversity_tsc.models import LITE
from diversity_tsc.training.losses import feature_diversity_penalty

logger = logging.getLogger(__name__)


__all__ = ["TrainerConfig", "TrainingHistory", "Trainer"]


@dataclass
class TrainerConfig:
    """Hyperparameters and behaviour switches for :class:`Trainer`."""

    epochs: int = 1500
    lr: float = 1e-3
    min_lr: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 50
    ce_weight: float = 0.8
    diversity_weight: float = 0.2
    # Only consider checkpointing after this epoch when co-training; matches
    # the published behaviour where the best model is selected late in training.
    best_model_min_epoch: int = 1100
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class TrainingHistory:
    """Per-epoch training/validation logs returned by :meth:`Trainer.fit`."""

    train_loss: List[float] = field(default_factory=list)
    train_ce_loss: List[float] = field(default_factory=list)
    train_div_loss: List[float] = field(default_factory=list)
    train_acc: List[float] = field(default_factory=list)
    val_loss: List[float] = field(default_factory=list)
    val_acc: List[float] = field(default_factory=list)
    learning_rate: List[float] = field(default_factory=list)
    best_epoch: int = 0
    best_train_loss: float = float("inf")
    duration_seconds: float = 0.0

    def to_dataframe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "loss": self.train_loss,
                "ce_loss": self.train_ce_loss,
                "div_loss": self.train_div_loss,
                "train_acc": self.train_acc,
                "val_loss": self.val_loss,
                "val_acc": self.val_acc,
                "learning_rate": self.learning_rate,
            }
        )


class Trainer:
    """Training loop for a single LITE model, optionally with reference models.

    When ``reference_models`` is empty the trainer optimises plain cross-entropy
    (baseline). When one or more frozen reference models are supplied, an
    additional feature-diversity penalty is added to the loss.
    """

    def __init__(
        self,
        model: LITE,
        train_loader: DataLoader,
        val_loader: DataLoader,
        config: TrainerConfig,
        reference_models: Optional[Sequence[LITE]] = None,
    ) -> None:
        self.model = model.to(config.device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config
        self.reference_models = [m.to(config.device).eval() for m in (reference_models or [])]
        for ref in self.reference_models:
            for p in ref.parameters():
                p.requires_grad_(False)

        self.criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer,
            mode="min",
            factor=config.scheduler_factor,
            patience=config.scheduler_patience,
            min_lr=config.min_lr,
        )

    # ------------------------------------------------------------------ private

    def _train_one_epoch(self) -> Tuple[float, float, float, float]:
        self.model.train()
        for ref in self.reference_models:
            ref.eval()

        device = self.config.device
        total_loss = total_ce = total_div = 0.0
        correct = total = 0

        for inputs, targets in self.train_loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)

            logits, feats = self.model.extract_features(inputs)
            ce_loss = self.criterion(logits, targets)

            if self.reference_models:
                ref_feats = []
                with torch.no_grad():
                    for ref in self.reference_models:
                        _, rf = ref.extract_features(inputs)
                        ref_feats.append(rf[-1])
                div_loss = feature_diversity_penalty(feats[-1], ref_feats)
                loss = self.config.ce_weight * ce_loss + self.config.diversity_weight * div_loss
            else:
                div_loss = torch.zeros((), device=device)
                loss = ce_loss

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            total_ce += ce_loss.item()
            total_div += div_loss.item() if div_loss.requires_grad or div_loss.numel() else 0.0

            preds = logits.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

        n_batches = max(len(self.train_loader), 1)
        return (
            total_loss / n_batches,
            total_ce / n_batches,
            total_div / n_batches,
            100.0 * correct / max(total, 1),
        )

    @torch.no_grad()
    def evaluate(self, loader: Optional[DataLoader] = None) -> Tuple[float, float]:
        """Compute average loss and accuracy on the given loader (val by default)."""
        loader = loader if loader is not None else self.val_loader
        self.model.eval()
        device = self.config.device

        total_loss = 0.0
        correct = total = 0
        for inputs, targets in loader:
            inputs = inputs.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            logits, _ = self.model(inputs)
            total_loss += self.criterion(logits, targets).item()
            preds = logits.argmax(dim=1)
            total += targets.size(0)
            correct += (preds == targets).sum().item()

        n_batches = max(len(loader), 1)
        return total_loss / n_batches, 100.0 * correct / max(total, 1)

    # ------------------------------------------------------------------- public

    def fit(self, output_directory: Optional[Path] = None) -> Tuple[TrainingHistory, dict]:
        """Run the full training schedule.

        Returns
        -------
        history:
            Per-epoch training/validation logs.
        best_state:
            ``state_dict`` of the model with the lowest cross-entropy loss seen
            during training (subject to ``best_model_min_epoch`` when
            co-training).
        """
        history = TrainingHistory()
        best_state = copy.deepcopy(self.model.state_dict())
        best_loss = float("inf")
        # Co-trained models in the original paper only start checkpointing late.
        min_epoch_for_best = (
            self.config.best_model_min_epoch if self.reference_models else 0
        )

        start = time.perf_counter()
        for epoch in range(self.config.epochs):
            train_loss, ce_loss, div_loss, train_acc = self._train_one_epoch()
            val_loss, val_acc = self.evaluate()

            self.scheduler.step(ce_loss if self.reference_models else train_loss)
            lr = self.optimizer.param_groups[0]["lr"]

            history.train_loss.append(train_loss)
            history.train_ce_loss.append(ce_loss)
            history.train_div_loss.append(div_loss)
            history.train_acc.append(train_acc)
            history.val_loss.append(val_loss)
            history.val_acc.append(val_acc)
            history.learning_rate.append(lr)

            if ce_loss <= best_loss and epoch >= min_epoch_for_best:
                best_loss = ce_loss
                best_state = copy.deepcopy(self.model.state_dict())
                history.best_epoch = epoch

            if epoch % 50 == 0 or epoch == self.config.epochs - 1:
                logger.info(
                    "epoch=%d ce=%.4f div=%.4f train_acc=%.2f val_acc=%.2f lr=%.2e",
                    epoch,
                    ce_loss,
                    div_loss,
                    train_acc,
                    val_acc,
                    lr,
                )

        history.duration_seconds = time.perf_counter() - start
        history.best_train_loss = best_loss

        if output_directory is not None:
            output_directory = Path(output_directory)
            output_directory.mkdir(parents=True, exist_ok=True)
            torch.save(best_state, output_directory / "best_model.pt")
            torch.save(self.model.state_dict(), output_directory / "last_model.pt")
            history.to_dataframe().to_csv(output_directory / "history.csv", index=False)

        return history, best_state
