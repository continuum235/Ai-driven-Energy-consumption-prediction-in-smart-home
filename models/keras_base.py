"""
Keras-model mixin that factors out boilerplate shared by LSTM, BiLSTM, CNN-LSTM,
and LEF-BiLSTM (build, compile, callbacks, fit, save/load, summary).
"""

from __future__ import annotations

import io
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

from models.base import BaseModel
from config.settings import Config

logger = logging.getLogger(__name__)


class KerasBaseModel(BaseModel):
    """Base class for all Keras / TensorFlow models."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.model = None  # built by subclass
        self.history: Dict[str, Any] = {}

    # ── callbacks factory ────────────────────────────────────────────────

    def _build_callbacks(self, checkpoint_path: str) -> List:
        from tensorflow.keras.callbacks import (  # noqa: E402
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
            TensorBoard,
        )

        cbs = [
            EarlyStopping(
                monitor="val_loss",
                patience=self.cfg.EARLY_STOPPING_PATIENCE,
                restore_best_weights=True,
            ),
            ModelCheckpoint(
                checkpoint_path,
                monitor="val_loss",
                save_best_only=True,
            ),
            ReduceLROnPlateau(
                monitor="val_loss",
                factor=self.cfg.REDUCE_LR_FACTOR,
                patience=self.cfg.REDUCE_LR_PATIENCE,
                min_lr=self.cfg.MIN_LR,
            ),
            TensorBoard(
                log_dir=str(self.cfg.LOGS_DIR / "tensorboard" / self.name),
                histogram_freq=1,
            ),
        ]
        return cbs

    # ── train ────────────────────────────────────────────────────────────

    def train(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: Optional[NDArray] = None,
        y_val: Optional[NDArray] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        assert self.model is not None, "Call build() before train()."

        checkpoint_path = str(
            self.cfg.MODELS_DIR / f"best_{self.name}.keras"
        )
        callbacks = self._build_callbacks(checkpoint_path)

        val_data = (X_val, y_val) if X_val is not None else None
        hist = self.model.fit(
            X_train,
            y_train,
            validation_data=val_data,
            epochs=self.cfg.EPOCHS,
            batch_size=self.cfg.BATCH_SIZE,
            callbacks=callbacks,
            verbose=1,
        )
        self.history = hist.history
        return self.history

    # ── predict ──────────────────────────────────────────────────────────

    def predict(self, X: NDArray) -> NDArray:
        return self.model.predict(X, verbose=0)

    # ── save / load ──────────────────────────────────────────────────────

    def save(self, path: str) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.model.save(path)
        logger.info("[%s] Model saved to %s", self.name, path)

    def load(self, path: str) -> None:
        import tensorflow as tf  # noqa: E402

        self.model = tf.keras.models.load_model(path)
        logger.info("[%s] Model loaded from %s", self.name, path)

    # ── summary ──────────────────────────────────────────────────────────

    def get_summary(self) -> str:
        if self.model is None:
            return f"{self.name} — model not built yet."
        buf = io.StringIO()
        self.model.summary(print_fn=lambda x: buf.write(x + "\n"))
        return buf.getvalue()
