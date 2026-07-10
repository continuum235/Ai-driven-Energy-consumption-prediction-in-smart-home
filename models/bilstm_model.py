"""
Standard Bidirectional LSTM model for energy consumption prediction.

This is the existing model from the original codebase, refactored into the
modular architecture.

Architecture:
    Bidirectional(LSTM(32, return_sequences=True)) → Dropout(0.2) →
    Bidirectional(LSTM(32)) → Dropout(0.2) →
    Dense(16, relu) → Dense(1)
"""

from __future__ import annotations

import logging

from config.settings import Config
from models.keras_base import KerasBaseModel

logger = logging.getLogger(__name__)


class BiLSTMModel(KerasBaseModel):
    """Two-layer Bidirectional LSTM — the original model from the project."""

    name: str = "BiLSTM"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._build()

    def _build(self) -> None:
        from tensorflow.keras.models import Sequential  # noqa: E402
        from tensorflow.keras.layers import (  # noqa: E402
            Bidirectional,
            Dense,
            Dropout,
            LSTM,
        )

        self.model = Sequential(
            [
                Bidirectional(
                    LSTM(self.cfg.LSTM_UNITS, return_sequences=True),
                    input_shape=(self.cfg.TIME_STEP, 1),
                ),
                Dropout(self.cfg.DROPOUT_RATE),
                Bidirectional(LSTM(self.cfg.LSTM_UNITS)),
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
