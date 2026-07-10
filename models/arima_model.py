"""
ARIMA baseline model for energy consumption prediction.

Since ARIMA is a statistical model (not deep learning), it does not use
Keras — it wraps `statsmodels.tsa.arima.model.ARIMA` and conforms to the
same BaseModel interface.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
from numpy.typing import NDArray

from config.settings import Config
from models.base import BaseModel

logger = logging.getLogger(__name__)


class ARIMAModel(BaseModel):
    """ARIMA wrapper with the shared train/predict/evaluate/save/load API."""

    name: str = "ARIMA"

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.order = cfg.ARIMA_ORDER
        self.fitted_model = None
        self.train_series: Optional[NDArray] = None

    # ── train ────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Fit ARIMA on the raw training series (flattened y_train)."""
        from statsmodels.tsa.arima.model import ARIMA  # noqa: E402

        series = y_train.flatten()[: self.cfg.ARIMA_MAX_SAMPLES]
        self.train_series = series
        logger.info(
            "Fitting ARIMA%s on %d samples …", self.order, len(series)
        )
        model = ARIMA(series, order=self.order)
        self.fitted_model = model.fit()
        logger.info("ARIMA fit complete. AIC=%.2f", self.fitted_model.aic)
        return {"aic": self.fitted_model.aic}

    # ── predict ──────────────────────────────────────────────────────────

    def predict(self, X: NDArray) -> NDArray:
        """
        Produce one-step-ahead forecasts for *len(X)* steps beyond the
        training period by using forecast().
        """
        n_steps = len(X)
        forecast = self.fitted_model.forecast(steps=n_steps)
        return np.asarray(forecast).reshape(-1, 1)

    # ── save / load ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.fitted_model.save(path)
        logger.info("[%s] Model saved to %s", self.name, path)

    def load(self, path: str) -> None:
        from statsmodels.tsa.arima.model import ARIMAResults  # noqa: E402

        self.fitted_model = ARIMAResults.load(path)
        logger.info("[%s] Model loaded from %s", self.name, path)
