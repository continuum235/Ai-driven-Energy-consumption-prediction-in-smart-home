"""
Abstract base class that every model (Keras-based *and* statistical) must implement.

Enforces a consistent interface: train, predict, evaluate, save, load.
"""

from __future__ import annotations

import abc
import logging
from typing import Any, Dict, Optional, Tuple

import numpy as np
from numpy.typing import NDArray
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

logger = logging.getLogger(__name__)


class BaseModel(abc.ABC):
    """Abstract model interface used by the training and evaluation modules."""

    name: str = "BaseModel"

    # ── core API ─────────────────────────────────────────────────────────

    @abc.abstractmethod
    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Train the model and return a history dict (may be empty for ARIMA)."""

    @abc.abstractmethod
    def predict(self, X: NDArray) -> NDArray:
        """Return predictions as a 1-D numpy array."""

    @abc.abstractmethod
    def save(self, path: str) -> None:
        """Persist the model to *path*."""

    @abc.abstractmethod
    def load(self, path: str) -> None:
        """Load a previously saved model from *path*."""

    # ── shared evaluation logic ──────────────────────────────────────────

    def evaluate(
        self, y_true: NDArray, y_pred: NDArray
    ) -> Dict[str, float]:
        """Compute MAE, RMSE, and R² on original-scale values."""
        y_true = np.asarray(y_true).flatten()
        y_pred = np.asarray(y_pred).flatten()

        mae = float(mean_absolute_error(y_true, y_pred))
        rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
        r2 = float(r2_score(y_true, y_pred))

        logger.info(
            "[%s] MAE=%.4f  RMSE=%.4f  R²=%.4f",
            self.name,
            mae,
            rmse,
            r2,
        )
        return {"MAE": mae, "RMSE": rmse, "R2": r2}

    def get_summary(self) -> str:
        """Return a human-readable summary string (overridden by Keras models)."""
        return f"{self.name} — no Keras summary available."
