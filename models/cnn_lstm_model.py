"""
CNN-LSTM hybrid model for energy consumption prediction.

Architecture:
    Conv1D(64, kernel=3, relu) → MaxPooling1D(2) →
    LSTM(32) → Dropout(0.2) →
    Dense(16, relu) → Dense(1)

The convolutional layer extracts local temporal features before passing
them to the LSTM for sequential modelling.
"""

from __future__ import annotations

import logging

from config.settings import Config
from models.keras_base import KerasBaseModel

logger = logging.getLogger(__name__)


class CNNLSTMModel(KerasBaseModel):
    """CNN-LSTM hybrid architecture."""

    name: str = "CNN-LSTM"

    def __init__(self, cfg: Config) -> None:
        super().__init__(cfg)
        self._build()

    def _build(self) -> None:
        from tensorflow.keras.models import Sequential  # noqa: E402
        from tensorflow.keras.layers import (  # noqa: E402
            Conv1D,
            Dense,
            Dropout,
            LSTM,
            MaxPooling1D,
        )

        self.model = Sequential(
            [
                Conv1D(
                    self.cfg.CNN_FILTERS,
                    kernel_size=self.cfg.CNN_KERNEL_SIZE,
                    activation="relu",
                    input_shape=(self.cfg.TIME_STEP, 1),
                ),
                MaxPooling1D(pool_size=2),
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
