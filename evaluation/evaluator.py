"""
Model evaluation utilities.

- Compute MAE / RMSE / R²
- Save predictions CSV (Actual, Predicted, Absolute Error, Squared Error)
- Save metrics CSV
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from config.settings import Config
from models.base import BaseModel

logger = logging.getLogger(__name__)


def evaluate_model(
    model: BaseModel,
    X_test: NDArray,
    y_test: NDArray,
    scaler: MinMaxScaler,
) -> Dict[str, float]:
    """
    Run inference, inverse-transform, and compute metrics.

    Returns dict with MAE, RMSE, R², Inference_Time_ms.
    """
    # Measure inference time
    start = time.perf_counter()
    y_pred_scaled = model.predict(X_test)
    inference_time = (time.perf_counter() - start) * 1000  # ms

    # Inverse-transform
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    metrics = model.evaluate(y_true, y_pred)
    metrics["Inference_Time_ms"] = round(inference_time, 4)

    return metrics


def save_predictions(
    model: BaseModel,
    X_test: NDArray,
    y_test: NDArray,
    scaler: MinMaxScaler,
    path: Path,
) -> None:
    """Save predictions CSV: Actual, Predicted, AbsoluteError, SquaredError."""
    y_pred_scaled = model.predict(X_test)
    y_pred = scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
    y_true = scaler.inverse_transform(y_test.reshape(-1, 1)).flatten()

    abs_err = np.abs(y_true - y_pred)
    sq_err = (y_true - y_pred) ** 2

    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Actual", "Predicted", "Absolute_Error", "Squared_Error"])
        for a, p, ae, se in zip(y_true, y_pred, abs_err, sq_err):
            writer.writerow([round(a, 6), round(p, 6), round(ae, 6), round(se, 6)])

    logger.info("[%s] Predictions saved to %s", model.name, path)


def save_metrics(
    results: Dict[str, Dict[str, float]], path: Path
) -> None:
    """Save a combined metrics CSV for all models."""
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = ["Model", "MAE", "RMSE", "R2", "Latency_ms", "Inference_Time_ms", "Training_Time_s"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, metrics in results.items():
            row = {"Model": model_name}
            row.update(metrics)
            writer.writerow(row)

    logger.info("Combined metrics saved to %s", path)
