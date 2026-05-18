#!/usr/bin/env python
"""Run a full diversity-driven ensemble for one or more datasets.

This script chains base training and ``ensemble_size - 1`` co-training stages,
wiring up reference checkpoints automatically. It is the recommended way to
reproduce the experiments from the paper.

Example
-------
.. code-block:: console

    python scripts/run_ensemble.py \\
        --datasets ECGFiveDays Coffee \\
        --ensemble-size 5 \\
        --seeds 37 5639 8976 8859 9876 \\
        --output-directory runs/
"""

from __future__ import annotations

import argparse
import logging
import subprocess
import sys
from pathlib import Path
from typing import List

logger = logging.getLogger("run_ensemble")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--datasets", nargs="+", required=True)
    parser.add_argument("--ensemble-size", type=int, default=5)
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        required=True,
        help="One seed per ensemble member.",
    )
    parser.add_argument("--output-directory", type=Path, default=Path("runs"))
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--ucr-root", type=Path, default=None)
    parser.add_argument(
        "--python",
        default=sys.executable,
        help="Python interpreter used to launch each training run.",
    )
    args = parser.parse_args()
    if len(args.seeds) != args.ensemble_size:
        parser.error(
            f"--seeds expects {args.ensemble_size} values, got {len(args.seeds)}."
        )
    return args


def _train_call(args: argparse.Namespace, ensemble_idx: int, seed: int, dataset: str,
                refs: List[Path]) -> list[str]:
    run_name = (
        f"base_seed_{seed}" if ensemble_idx == 1 else f"co{ensemble_idx}_seed_{seed}"
    )
    cmd = [
        args.python, "-m", "diversity_tsc.cli", "train",
        "--datasets", dataset,
        "--ensemble-size", str(ensemble_idx),
        "--seed", str(seed),
        "--epochs", str(args.epochs),
        "--output-directory", str(args.output_directory),
        "--run-name", run_name,
    ]
    if args.ucr_root is not None:
        cmd += ["--ucr-root", str(args.ucr_root)]
    if refs:
        cmd += ["--reference-checkpoints", *[str(p) for p in refs]]
    return cmd


def main() -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s")
    args = _parse_args()

    for dataset in args.datasets:
        refs: List[Path] = []
        for idx, seed in enumerate(args.seeds, start=1):
            cmd = _train_call(args, idx, seed, dataset, refs)
            logger.info("[%s | member %d/%d] %s", dataset, idx, args.ensemble_size, " ".join(cmd))
            subprocess.run(cmd, check=True)
            run_name = (
                f"base_seed_{seed}" if idx == 1 else f"co{idx}_seed_{seed}"
            )
            refs.append(args.output_directory / run_name / dataset / "best_model.pt")
    return 0


if __name__ == "__main__":
    sys.exit(main())
