"""Shared utilities: seeding, logging, and plotting helpers."""

from __future__ import annotations

import logging
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Optional

import numpy as np
import torch

__all__ = ["set_seed", "configure_logging", "plot_curves", "UCR_DATASETS"]


def set_seed(seed: int, deterministic: bool = True) -> None:
    """Seed Python, NumPy and PyTorch RNGs (CPU + CUDA)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def configure_logging(level: int = logging.INFO, log_file: Optional[Path] = None) -> None:
    """Configure root logger with a sensible format. Idempotent."""
    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))

    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
        handlers=handlers,
        force=True,
    )


def plot_curves(
    train_losses: Iterable[float],
    val_losses: Iterable[float],
    train_accs: Iterable[float],
    val_accs: Iterable[float],
    output_directory: Path,
) -> None:
    """Save ``losses.png`` and ``accuracies.png`` to ``output_directory``.

    Matplotlib is imported lazily so the package can be used in environments
    without a display backend.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_directory = Path(output_directory)
    output_directory.mkdir(parents=True, exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(list(train_losses), label="train_loss")
    ax.plot(list(val_losses), label="val_loss")
    ax.set_xlabel("epoch")
    ax.set_ylabel("loss")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "losses.png", dpi=120)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(list(train_accs), label="train_acc")
    ax.plot(list(val_accs), label="val_acc")
    ax.set_xlabel("epoch")
    ax.set_ylabel("accuracy (%)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_directory / "accuracies.png", dpi=120)
    plt.close(fig)


# fmt: off
UCR_DATASETS: tuple[str, ...] = (
    "ACSF1", "Adiac", "AllGestureWiimoteX", "AllGestureWiimoteY", "AllGestureWiimoteZ",
    "ArrowHead", "Beef", "BeetleFly", "BirdChicken", "BME", "Car", "CBF", "Chinatown",
    "ChlorineConcentration", "CinCECGTorso", "Coffee", "Computers", "CricketX",
    "CricketY", "CricketZ", "Crop", "DiatomSizeReduction",
    "DistalPhalanxOutlineAgeGroup", "DistalPhalanxOutlineCorrect", "DistalPhalanxTW",
    "DodgerLoopDay", "DodgerLoopGame", "DodgerLoopWeekend", "Earthquakes", "ECG200",
    "ECG5000", "ECGFiveDays", "ElectricDevices", "EOGHorizontalSignal",
    "EOGVerticalSignal", "EthanolLevel", "FaceAll", "FaceFour", "FacesUCR",
    "FiftyWords", "Fish", "FordA", "FordB", "FreezerRegularTrain",
    "FreezerSmallTrain", "Fungi", "GestureMidAirD1", "GestureMidAirD2",
    "GestureMidAirD3", "GesturePebbleZ1", "GesturePebbleZ2", "GunPoint",
    "GunPointAgeSpan", "GunPointMaleVersusFemale", "GunPointOldVersusYoung",
    "Ham", "HandOutlines", "Haptics", "Herring", "HouseTwenty", "InlineSkate",
    "InsectEPGRegularTrain", "InsectEPGSmallTrain", "InsectWingbeatSound",
    "ItalyPowerDemand", "LargeKitchenAppliances", "Lightning2", "Lightning7",
    "Mallat", "Meat", "MedicalImages", "MelbournePedestrian",
    "MiddlePhalanxOutlineAgeGroup", "MiddlePhalanxOutlineCorrect",
    "MiddlePhalanxTW", "MixedShapesRegularTrain", "MixedShapesSmallTrain",
    "MoteStrain", "NonInvasiveFetalECGThorax1", "NonInvasiveFetalECGThorax2",
    "OliveOil", "OSULeaf", "PhalangesOutlinesCorrect", "Phoneme",
    "PickupGestureWiimoteZ", "PigAirwayPressure", "PigArtPressure", "PigCVP",
    "PLAID", "Plane", "PowerCons", "ProximalPhalanxOutlineAgeGroup",
    "ProximalPhalanxOutlineCorrect", "ProximalPhalanxTW", "RefrigerationDevices",
    "Rock", "ScreenType", "SemgHandGenderCh2", "SemgHandMovementCh2",
    "SemgHandSubjectCh2", "ShakeGestureWiimoteZ", "ShapeletSim", "ShapesAll",
    "SmallKitchenAppliances", "SmoothSubspace", "SonyAIBORobotSurface1",
    "SonyAIBORobotSurface2", "StarLightCurves", "Strawberry", "SwedishLeaf",
    "Symbols", "SyntheticControl", "ToeSegmentation1", "ToeSegmentation2", "Trace",
    "TwoLeadECG", "TwoPatterns", "UMD", "UWaveGestureLibraryAll",
    "UWaveGestureLibraryX", "UWaveGestureLibraryY", "UWaveGestureLibraryZ",
    "Wafer", "Wine", "WordSynonyms", "Worms", "WormsTwoClass", "Yoga",
)
# fmt: on
