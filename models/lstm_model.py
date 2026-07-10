"""
Standard (unidirectional) LSTM model for energy consumption prediction.

Architecture:
    LSTM(32, return_sequences=True) → Dropout(0.2) →
    LSTM(32) → Dropout(0.2) →
    Dense(16, relu) → Dense(1)
"""

from __future__ import annotations

import logging

from config.settings import Config
from models.keras_base import KerasBaseModel

logger = logging.getLogger(__name__)


class LSTMModel(KerasBaseModel):
    """Two-layer unidirectional LSTM baseline."""

    name: str = "LSTM"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._build()

    def _build(self) -> None:
        from tensorflow.keras.models import Sequential  # noqa: E402
        from tensorflow.keras.layers import LSTM, Dense, Dropout  # noqa: E402

        self.model = Sequential(
            [
                LSTM(
                    self.cfg.LSTM_UNITS,
                    return_sequences=True,
                    input_shape=(self.cfg.TIME_STEP, 1),
                ),
                Dropout(self.cfg.DROPOUT_RATE),
                LSTM(self.cfg.LSTM_UNITS),
                Dropout(self.cfg.DROPOUT_RATE),
                Dense(self.cfg.DENSE_UNITS, activation="relu"),
                Dense(1),
            ]
        )
        self.model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mae"],
        )
        logger.info("[%s] Model built.", self.name)
