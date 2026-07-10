"""
Model registry — maps model names to their classes for easy lookup from
configuration or CLI arguments.
"""

from __future__ import annotations

from typing import Dict, Type

from config.settings import Config
from models.base import BaseModel
from models.arima_model import ARIMAModel
from models.lstm_model import LSTMModel
from models.cnn_lstm_model import CNNLSTMModel
from models.bilstm_model import BiLSTMModel
from models.lef_bilstm_model import LEFBiLSTMModel

MODEL_REGISTRY: Dict[str, Type[BaseModel]] = {
    "ARIMA": ARIMAModel,
    "LSTM": LSTMModel,
    "CNN-LSTM": CNNLSTMModel,
    "BiLSTM": BiLSTMModel,
    "LEF-BiLSTM": LEFBiLSTMModel,
}


def build_model(name: str, cfg: Config) -> BaseModel:
    """Instantiate a model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(
            f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY.keys())}"
        )
    return MODEL_REGISTRY[name](cfg)
