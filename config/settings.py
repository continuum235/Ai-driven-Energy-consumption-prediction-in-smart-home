"""
Centralized configuration for the AI-Driven Energy Consumption Prediction project.

All hyperparameters, paths, and constants are defined here so that every
module in the repository draws from a single source of truth.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import List


@dataclass
class Config:
    """Project-wide configuration container."""

    # ── Reproducibility ──────────────────────────────────────────────────
    SEED: int = 42

    # ── Paths ────────────────────────────────────────────────────────────
    BASE_DIR: Path = field(default_factory=lambda: Path(os.path.dirname(os.path.abspath(__file__))).parent)
    DATA_FILENAME: str = "data.txt"

    @property
    def DATA_PATH(self) -> Path:
        return self.BASE_DIR / "dataset" / self.DATA_FILENAME

    @property
    def OUTPUTS_DIR(self) -> Path:
        return self.BASE_DIR / "outputs"

    @property
    def METRICS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "metrics"

    @property
    def MODELS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "models"

    @property
    def FIGURES_DIR(self) -> Path:
        return self.BASE_DIR / "figures"

    @property
    def REPORTS_DIR(self) -> Path:
        return self.BASE_DIR / "reports"

    @property
    def LOGS_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "logs"

    @property
    def TFLITE_DIR(self) -> Path:
        return self.OUTPUTS_DIR / "tflite"

    # ── Dataset ──────────────────────────────────────────────────────────
    NUM_ROWS: int = 20_000
    TARGET_COLUMN: str = "Global_active_power"
    DATE_FORMAT: str = "%d/%m/%Y %H:%M:%S"
    SEPARATOR: str = ";"

    # ── Preprocessing ────────────────────────────────────────────────────
    TIME_STEP: int = 30  # sliding window size
    TRAIN_RATIO: float = 0.70
    VAL_RATIO: float = 0.15  # test is 1 - train - val

    # ── Training ─────────────────────────────────────────────────────────
    EPOCHS: int = 30
    BATCH_SIZE: int = 32
    LEARNING_RATE: float = 0.001
    EARLY_STOPPING_PATIENCE: int = 5
    REDUCE_LR_PATIENCE: int = 3
    REDUCE_LR_FACTOR: float = 0.5
    MIN_LR: float = 1e-6

    # ── Model architecture ───────────────────────────────────────────────
    LSTM_UNITS: int = 32
    DENSE_UNITS: int = 16
    DROPOUT_RATE: float = 0.2
    CNN_FILTERS: int = 64
    CNN_KERNEL_SIZE: int = 3

    # ── ARIMA ────────────────────────────────────────────────────────────
    ARIMA_ORDER: tuple = (5, 1, 0)
    ARIMA_MAX_SAMPLES: int = 3_000  # ARIMA is O(n^3); cap training size

    # ── Edge Computing ───────────────────────────────────────────────────
    EDGE_NUM_INFERENCE_RUNS: int = 100  # for latency measurement
    MQTT_BROKER: str = "localhost"
    MQTT_PORT: int = 1883
    MQTT_TOPIC: str = "smart_home/energy"

    # ── Federated Learning ───────────────────────────────────────────────
    NUM_CLIENTS: int = 3
    NUM_FED_ROUNDS: int = 5
    FED_LOCAL_EPOCHS: int = 3

    # ── Model Registry ───────────────────────────────────────────────────
    MODEL_NAMES: List[str] = field(
        default_factory=lambda: [
            "ARIMA",
            "LSTM",
            "CNN-LSTM",
            "BiLSTM",
            "LEF-BiLSTM",
        ]
    )

    def ensure_dirs(self) -> None:
        """Create all output directories if they don't exist."""
        for d in [
            self.OUTPUTS_DIR,
            self.METRICS_DIR,
            self.MODELS_DIR,
            self.FIGURES_DIR,
            self.REPORTS_DIR,
            self.LOGS_DIR,
            self.TFLITE_DIR,
        ]:
            d.mkdir(parents=True, exist_ok=True)
