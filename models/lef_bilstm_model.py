"""
LEF-BiLSTM — Lightweight Edge-Friendly Bidirectional LSTM.

This is the architecture described in the research paper:
"AI-Driven Energy Consumption Prediction in Smart Homes using IoT and Edge Computing"

Design principles for edge deployment:
    • Reduced LSTM units (16 vs 32) to cut parameter count
    • Single Bidirectional LSTM layer (instead of stacked two)
    • Smaller dense head (8 units)
    • L2 kernel regularisation for better generalisation on constrained data
    • Lower dropout to compensate for the reduced capacity

Architecture:
    Bidirectional(LSTM(16, kernel_regularizer=l2)) → Dropout(0.1) →
    Dense(8, relu) → Dense(1)

This trades a small amount of accuracy for a ~4× reduction in parameters
and ~2–3× faster inference, making it deployable on IoT edge gateways.
"""

from __future__ import annotations

import logging

from config.settings import Config
from models.keras_base import KerasBaseModel

logger = logging.getLogger(__name__)


class LEFBiLSTMModel(KerasBaseModel):
    """Lightweight Edge-Friendly BiLSTM for edge / IoT deployment."""

    name: str = "LEF-BiLSTM"

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
        from tensorflow.keras.regularizers import l2  # noqa: E402

        lef_units = self.cfg.LSTM_UNITS // 2  # 16
        lef_dense = self.cfg.DENSE_UNITS // 2  # 8
        lef_dropout = self.cfg.DROPOUT_RATE / 2  # 0.1

        self.model = Sequential(
            [
                Bidirectional(
                    LSTM(
                        lef_units,
                        kernel_regularizer=l2(1e-4),
                    ),
                    input_shape=(self.cfg.TIME_STEP, 1),
                ),
                Dropout(lef_dropout),
                Dense(lef_dense, activation="relu"),
                Dense(1),
            ]
        )
        self.model.compile(
            loss="mean_squared_error",
            optimizer="adam",
            metrics=["mae"],
        )
        logger.info(
            "[%s] Model built — lightweight edge architecture (%d LSTM units, %d dense units).",
            self.name,
            lef_units,
            lef_dense,
        )
