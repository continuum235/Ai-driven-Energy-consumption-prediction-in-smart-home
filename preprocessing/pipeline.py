"""
Preprocessing pipeline for the UCI Household Electric Power Consumption dataset.

Stages:
    1. Dataset loading (semicolon-separated text file)
    2. Datetime conversion (Date + Time columns)
    3. Numeric conversion (handles '?' as missing values)
    4. Missing value removal
    5. MinMax normalisation (fit on training data only)
    6. Sliding window generation
    7. Chronological Train / Validation / Test split
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Tuple

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.preprocessing import MinMaxScaler

from config.settings import Config

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container holding all preprocessed arrays and the fitted scaler."""

    X_train: NDArray[np.float32]
    y_train: NDArray[np.float32]
    X_val: NDArray[np.float32]
    y_val: NDArray[np.float32]
    X_test: NDArray[np.float32]
    y_test: NDArray[np.float32]
    scaler: MinMaxScaler
    raw_df: pd.DataFrame


# ── Step-by-step helpers ─────────────────────────────────────────────────


def load_dataset(cfg: Config) -> pd.DataFrame:
    """Load the raw semicolon-separated dataset."""
    logger.info("Loading dataset from %s", cfg.DATA_PATH)
    df = pd.read_csv(cfg.DATA_PATH, sep=cfg.SEPARATOR, low_memory=False)
    df = df.head(cfg.NUM_ROWS)
    logger.info("Loaded %d rows", len(df))
    return df


def convert_datetime(df: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Combine Date + Time into a single datetime column."""
    df = df.copy()
    df["datetime"] = pd.to_datetime(
        df["Date"] + " " + df["Time"],
        format=cfg.DATE_FORMAT,
        errors="coerce",
    )
    return df


def convert_numeric(df: pd.DataFrame, target: str) -> pd.DataFrame:
    """Cast target column to numeric (handles '?' sentinel in the UCI data)."""
    df = df.copy()
    df[target] = pd.to_numeric(df[target], errors="coerce")
    return df


def remove_missing(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with NaN in any column and reset index."""
    n_before = len(df)
    df = df.dropna().reset_index(drop=True)
    n_after = len(df)
    logger.info(
        "Dropped %d rows with missing/invalid values (%d -> %d)",
        n_before - n_after,
        n_before,
        n_after,
    )
    return df


def normalise(
    train: NDArray, val: NDArray, test: NDArray
) -> Tuple[NDArray, NDArray, NDArray, MinMaxScaler]:
    """Fit MinMaxScaler on *train only* and transform all splits."""
    scaler = MinMaxScaler()
    train_scaled = scaler.fit_transform(train)
    val_scaled = scaler.transform(val)
    test_scaled = scaler.transform(test)
    return train_scaled, val_scaled, test_scaled, scaler


def create_sequences(
    data: NDArray, time_step: int = 30
) -> Tuple[NDArray, NDArray]:
    """Create sliding-window sequences for supervised learning."""
    X, Y = [], []
    for i in range(len(data) - time_step - 1):
        X.append(data[i : (i + time_step), 0])
        Y.append(data[i + time_step, 0])
    return np.array(X), np.array(Y)


def chronological_split(
    series: NDArray, train_ratio: float, val_ratio: float
) -> Tuple[NDArray, NDArray, NDArray]:
    """Split a 1-D series chronologically (before scaling to avoid leakage)."""
    n = len(series)
    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))
    return (
        series[:train_end].reshape(-1, 1),
        series[train_end:val_end].reshape(-1, 1),
        series[val_end:].reshape(-1, 1),
    )


# ── Main entry point ────────────────────────────────────────────────────


def load_and_preprocess(cfg: Config) -> PreprocessedData:
    """Execute the full preprocessing pipeline and return split arrays."""
    # 1. Load
    df = load_dataset(cfg)

    # 2. Datetime
    df = convert_datetime(df, cfg)

    # 3. Select target + convert
    df = df[["datetime", cfg.TARGET_COLUMN]]
    df = convert_numeric(df, cfg.TARGET_COLUMN)

    # 4. Missing values
    df = remove_missing(df)

    # 5. Chronological split (raw values)
    train_raw, val_raw, test_raw = chronological_split(
        df[cfg.TARGET_COLUMN].values, cfg.TRAIN_RATIO, cfg.VAL_RATIO
    )

    # 6. Normalise
    train_sc, val_sc, test_sc, scaler = normalise(train_raw, val_raw, test_raw)

    # 7. Sliding windows
    X_train, y_train = create_sequences(train_sc, cfg.TIME_STEP)
    X_val, y_val = create_sequences(val_sc, cfg.TIME_STEP)
    X_test, y_test = create_sequences(test_sc, cfg.TIME_STEP)

    # Reshape for LSTM: (samples, timesteps, 1)
    X_train = X_train.reshape(X_train.shape[0], X_train.shape[1], 1)
    X_val = X_val.reshape(X_val.shape[0], X_val.shape[1], 1)
    X_test = X_test.reshape(X_test.shape[0], X_test.shape[1], 1)

    logger.info(
        "Split sizes — Train: %s, Val: %s, Test: %s",
        X_train.shape,
        X_val.shape,
        X_test.shape,
    )

    return PreprocessedData(
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
        X_test=X_test,
        y_test=y_test,
        scaler=scaler,
        raw_df=df,
    )
