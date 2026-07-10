"""
Baseline comparison — trains all models on the same dataset and generates
comparison.csv with MAE, RMSE, R², Latency, and Inference Time.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from config.settings import Config
from models.base import BaseModel
from models.registry import build_model
from training.trainer import run_training

logger = logging.getLogger(__name__)


def run_comparison(
    X_train: NDArray,
    y_train: NDArray,
    X_val: NDArray,
    y_val: NDArray,
    X_test: NDArray,
    y_test: NDArray,
    scaler: MinMaxScaler,
    cfg: Config,
) -> Dict[str, Dict[str, float]]:
    """
    Train and evaluate every model in cfg.MODEL_NAMES.

    Returns {model_name: {MAE, RMSE, R2, Latency_ms, Inference_Time_ms}}.
    """
    results: Dict[str, Dict[str, float]] = {}

    for name in cfg.MODEL_NAMES:
        logger.info("=" * 60)
        logger.info("  BASELINE COMPARISON — %s", name)
        logger.info("=" * 60)

        model = build_model(name, cfg)

        # ── Train ────────────────────────────────────────────────────
        train_start = time.perf_counter()
        run_training(model, X_train, y_train, X_val, y_val, cfg)
        train_time = time.perf_counter() - train_start

        # ── Predict + Evaluate ───────────────────────────────────────
        infer_start = time.perf_counter()
        y_pred_scaled = model.predict(X_test)
        infer_time = (time.perf_counter() - infer_start) * 1000  # ms

        y_pred = scaler.inverse_transform(
            y_pred_scaled.reshape(-1, 1)
        ).flatten()
        y_true = scaler.inverse_transform(
            y_test.reshape(-1, 1)
        ).flatten()

        metrics = model.evaluate(y_true, y_pred)
        metrics["Latency_ms"] = round(infer_time / max(len(X_test), 1), 4)
        metrics["Inference_Time_ms"] = round(infer_time, 4)
        metrics["Training_Time_s"] = round(train_time, 2)

        results[name] = metrics

        # ── Save per-model predictions ───────────────────────────────
        from evaluation.evaluator import save_predictions

        pred_path = cfg.METRICS_DIR / f"{name}_predictions.csv"
        save_predictions(model, X_test, y_test, scaler, pred_path)

    # ── Save comparison CSV ──────────────────────────────────────────
    comp_path = cfg.METRICS_DIR / "comparison.csv"
    comp_path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "Model", "MAE", "RMSE", "R2",
        "Latency_ms", "Inference_Time_ms", "Training_Time_s",
    ]
    with open(comp_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for model_name, m in results.items():
            row = {"Model": model_name}
            row.update(m)
            writer.writerow(row)

    logger.info("Comparison results saved to %s", comp_path)
    return results
