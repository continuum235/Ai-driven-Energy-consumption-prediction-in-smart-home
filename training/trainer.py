"""
Reusable training driver that works with any BaseModel implementation.

Handles:
    - Seed setting
    - Model training (delegates to model.train())
    - History saving (CSV)
    - Model saving
    - Model summary export
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from config.settings import Config
from models.base import BaseModel

logger = logging.getLogger(__name__)


def _set_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import tensorflow as tf  # noqa: E402

    np.random.seed(seed)
    tf.random.set_seed(seed)


def save_training_history(
    history: Dict[str, Any], path: Path, model_name: str
) -> None:
    """Write training history to CSV (epoch, loss, val_loss, lr)."""
    if not history or "loss" not in history:
        logger.warning("[%s] No training history to save.", model_name)
        return

    path.parent.mkdir(parents=True, exist_ok=True)

    n_epochs = len(history["loss"])
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Epoch", "Loss", "Val_Loss", "Learning_Rate"])
        for i in range(n_epochs):
            lr = history.get("lr", [None] * n_epochs)[i]
            val_loss = history.get("val_loss", [None] * n_epochs)[i]
            writer.writerow([i + 1, history["loss"][i], val_loss, lr])

    logger.info("[%s] Training history saved to %s", model_name, path)


def save_model_summary(model: BaseModel, path: Path) -> None:
    """Write model.summary() to a text file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    summary_text = model.get_summary()
    with open(path, "w", encoding="utf-8") as f:
        f.write(f"Model: {model.name}\n")
        f.write("=" * 60 + "\n")
        f.write(summary_text)
    logger.info("[%s] Model summary saved to %s", model.name, path)


def run_training(
    model: BaseModel,
    X_train: NDArray,
    y_train: NDArray,
    X_val: Optional[NDArray],
    y_val: Optional[NDArray],
    cfg: Config,
) -> Dict[str, Any]:
    """
    Train a model, save its history, checkpoint, and summary.

    Returns the training history dict.
    """
    _set_seeds(cfg.SEED)

    logger.info("=" * 60)
    logger.info("Training model: %s", model.name)
    logger.info("=" * 60)

    # Check if model already exists
    model_path_keras = str(cfg.MODELS_DIR / f"{model.name}_final.keras")
    model_path_pkl = str(cfg.MODELS_DIR / f"{model.name}_final.pkl")
    
    if Path(model_path_keras).exists() or Path(model_path_pkl).exists():
        logger.info("[%s] Found existing trained model, skipping training.", model.name)
        if Path(model_path_keras).exists():
            model.load(model_path_keras)
        else:
            model.load(model_path_pkl)
        return {}

    start = time.perf_counter()
    history = model.train(X_train, y_train, X_val, y_val)
    elapsed = time.perf_counter() - start
    logger.info("[%s] Training completed in %.2f s", model.name, elapsed)

    # Save model
    model_path = str(cfg.MODELS_DIR / f"{model.name}_final.keras")
    try:
        model.save(model_path)
    except Exception:
        # ARIMA save uses a different extension
        model_path = str(cfg.MODELS_DIR / f"{model.name}_final.pkl")
        model.save(model_path)

    # Save history
    hist_path = cfg.METRICS_DIR / f"{model.name}_training_history.csv"
    save_training_history(history, hist_path, model.name)

    # Save summary
    summary_path = cfg.METRICS_DIR / f"{model.name}_model_summary.txt"
    save_model_summary(model, summary_path)

    return history
