"""Unified CLI for training base and co-trained LITE models.

This single entry point replaces the ``base.py`` and ``cotrain[_N].py`` scripts.

Examples
--------
Train a baseline model:

.. code-block:: console

    diversity-tsc train --ensemble-size 1 --datasets ECGFiveDays --seed 42

Train an ensemble member that decorrelates against two previously trained
checkpoints (``--ensemble-size`` here equals ``1 + len(--reference-checkpoints)``):

.. code-block:: console

    diversity-tsc train \\
        --ensemble-size 3 \\
        --datasets ECGFiveDays \\
        --seed 8976 \\
        --reference-checkpoints runs/base_seed_37/ECGFiveDays/best_model.pt \\
                                runs/co2_seed_5639/ECGFiveDays/best_model.pt
"""

from __future__ import annotations

import argparse
import logging
import sys
from collections.abc import Sequence
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import torch

from diversity_tsc import __version__
from diversity_tsc.data import load_ucr_dataset, make_dataloader
from diversity_tsc.models import LITE
from diversity_tsc.training import Trainer, TrainerConfig
from diversity_tsc.utils import UCR_DATASETS, configure_logging, plot_curves, set_seed

logger = logging.getLogger("diversity_tsc.cli")


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="diversity-tsc",
        description="Diversity-driven ensemble learning for time series classification.",
    )
    parser.add_argument("--version", action="version", version=f"%(prog)s {__version__}")

    sub = parser.add_subparsers(dest="command", required=True)
    train_p = sub.add_parser("train", help="Train a LITE model (baseline or co-trained).")

    train_p.add_argument(
        "--datasets",
        nargs="+",
        default=list(UCR_DATASETS),
        help="UCR dataset names to train on (default: all 128 datasets).",
    )
    train_p.add_argument(
        "--ucr-root",
        type=Path,
        default=None,
        help="Path to UCRArchive_2018. Falls back to $UCR_ARCHIVE_ROOT then "
        "~/.cache/ucr_archive/UCRArchive_2018.",
    )
    train_p.add_argument(
        "--output-directory",
        type=Path,
        default=Path("runs"),
        help="Parent directory for run outputs.",
    )
    train_p.add_argument(
        "--ensemble-size",
        type=int,
        default=1,
        help="Position of this model in the ensemble. 1 = baseline, "
        "N = decorrelated against (N-1) reference checkpoints.",
    )
    train_p.add_argument(
        "--reference-checkpoints",
        nargs="*",
        type=Path,
        default=[],
        help="Paths to (ensemble-size - 1) frozen reference checkpoints, in order.",
    )
    train_p.add_argument("--seed", type=int, default=42, help="Random seed.")
    train_p.add_argument("--epochs", type=int, default=1500)
    train_p.add_argument("--batch-size", type=int, default=64)
    train_p.add_argument("--lr", type=float, default=1e-3)
    train_p.add_argument("--ce-weight", type=float, default=0.8)
    train_p.add_argument("--diversity-weight", type=float, default=0.2)
    train_p.add_argument("--n-filters", type=int, default=32)
    train_p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cpu", "cuda"],
    )
    train_p.add_argument(
        "--run-name",
        default=None,
        help="Subdirectory under --output-directory. Defaults to a name "
        "derived from --ensemble-size and --seed.",
    )
    train_p.add_argument(
        "--log-level", default="INFO", choices=["DEBUG", "INFO", "WARNING", "ERROR"]
    )
    return parser


def _validate(args: argparse.Namespace) -> None:
    expected_refs = max(args.ensemble_size - 1, 0)
    if len(args.reference_checkpoints) != expected_refs:
        raise SystemExit(
            f"--ensemble-size {args.ensemble_size} expects {expected_refs} "
            f"reference checkpoint(s); got {len(args.reference_checkpoints)}."
        )
    for ckpt in args.reference_checkpoints:
        if not ckpt.is_file():
            raise SystemExit(f"Reference checkpoint not found: {ckpt}")


def _load_references(
    paths: Sequence[Path], length_TS: int, n_classes: int, n_filters: int, device: str
) -> List[LITE]:
    refs: List[LITE] = []
    for path in paths:
        model = LITE(length_TS=length_TS, n_classes=n_classes, n_filters=n_filters)
        state = torch.load(path, map_location=device)
        model.load_state_dict(state)
        refs.append(model)
        logger.info("Loaded reference checkpoint: %s", path)
    return refs


def _train_one(args: argparse.Namespace, dataset: str) -> None:
    xtrain, ytrain, xtest, ytest = load_ucr_dataset(dataset, args.ucr_root)
    length_TS = xtrain.shape[1]
    n_classes = len(np.unique(ytrain))
    logger.info(
        "dataset=%s | n_train=%d n_test=%d length=%d n_classes=%d",
        dataset,
        len(xtrain),
        len(xtest),
        length_TS,
        n_classes,
    )

    train_loader = make_dataloader(xtrain, ytrain, batch_size=args.batch_size, shuffle=True)
    val_loader = make_dataloader(xtest, ytest, batch_size=args.batch_size, shuffle=False)

    set_seed(args.seed)
    model = LITE(length_TS=length_TS, n_classes=n_classes, n_filters=args.n_filters)
    refs = _load_references(
        args.reference_checkpoints, length_TS, n_classes, args.n_filters, args.device
    )

    cfg = TrainerConfig(
        epochs=args.epochs,
        lr=args.lr,
        ce_weight=args.ce_weight,
        diversity_weight=args.diversity_weight,
        device=args.device,
    )
    trainer = Trainer(model, train_loader, val_loader, cfg, reference_models=refs)

    run_name = args.run_name or (
        f"base_seed_{args.seed}"
        if args.ensemble_size == 1
        else f"co{args.ensemble_size}_seed_{args.seed}"
    )
    out_dir = args.output_directory / run_name / dataset
    out_dir.mkdir(parents=True, exist_ok=True)

    history, best_state = trainer.fit(out_dir)

    # Evaluate the best model and write metrics
    model.load_state_dict(best_state)
    val_loss, val_acc = trainer.evaluate(val_loader)
    pd.DataFrame(
        {
            "train_loss": [history.best_train_loss],
            "test_accuracy": [val_acc],
            "test_loss": [val_loss],
            "duration_seconds": [history.duration_seconds],
            "best_epoch": [history.best_epoch],
        }
    ).to_csv(out_dir / "metrics.csv", index=False)

    plot_curves(
        history.train_loss, history.val_loss, history.train_acc, history.val_acc, out_dir
    )
    logger.info(
        "FINISHED %s -> test_acc=%.2f%% (best_epoch=%d, %.1fs)",
        dataset,
        val_acc,
        history.best_epoch,
        history.duration_seconds,
    )


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    configure_logging(level=getattr(logging, args.log_level))

    if args.command == "train":
        _validate(args)
        for dataset in args.datasets:
            try:
                _train_one(args, dataset)
            except FileNotFoundError as exc:
                logger.error("Skipping %s: %s", dataset, exc)
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
