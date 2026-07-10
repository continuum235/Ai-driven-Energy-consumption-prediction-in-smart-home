"""
Edge-inference latency measurement.

Measures Keras model inference latency AND TFLite inference latency,
then exports latency.csv for comparison.
"""

from __future__ import annotations

import csv
import logging
import time
from pathlib import Path
from typing import Dict, List

import numpy as np
from numpy.typing import NDArray

from config.settings import Config

logger = logging.getLogger(__name__)


def _measure_keras_latency(
    model_path: str, X_sample: NDArray, n_runs: int
) -> List[float]:
    """Measure per-sample inference latency for a Keras model."""
    import tensorflow as tf  # noqa: E402

    model = tf.keras.models.load_model(model_path)
    latencies = []
    for _ in range(n_runs):
        sample = X_sample[:1]
        start = time.perf_counter()
        model.predict(sample, verbose=0)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def _measure_tflite_latency(
    tflite_path: str, X_sample: NDArray, n_runs: int
) -> List[float]:
    """Measure per-sample inference latency for a TFLite model."""
    import tensorflow as tf  # noqa: E402

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    latencies = []
    for _ in range(n_runs):
        sample = X_sample[:1].astype(np.float32)
        start = time.perf_counter()
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        _ = interpreter.get_tensor(output_details[0]["index"])
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def measure_latency(
    keras_model_path: str,
    tflite_model_path: str,
    X_sample: NDArray,
    cfg: Config,
) -> Dict[str, float]:
    """
    Measure and compare Keras vs TFLite inference latency.

    Saves latency.csv and returns summary statistics.
    """
    n = cfg.EDGE_NUM_INFERENCE_RUNS
    logger.info("Measuring latency over %d runs …", n)

    keras_lat = _measure_keras_latency(keras_model_path, X_sample, n)
    tflite_lat = _measure_tflite_latency(tflite_model_path, X_sample, n)

    summary = {
        "Keras_Mean_ms": round(float(np.mean(keras_lat)), 4),
        "Keras_Std_ms": round(float(np.std(keras_lat)), 4),
        "TFLite_Mean_ms": round(float(np.mean(tflite_lat)), 4),
        "TFLite_Std_ms": round(float(np.std(tflite_lat)), 4),
        "Speedup": round(float(np.mean(keras_lat) / max(np.mean(tflite_lat), 1e-6)), 2),
    }

    # Save CSV
    csv_path = cfg.METRICS_DIR / "latency.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["Run", "Keras_ms", "TFLite_ms"])
        for i, (k, t) in enumerate(zip(keras_lat, tflite_lat)):
            writer.writerow([i + 1, round(k, 4), round(t, 4)])

    logger.info("Latency results: %s", summary)
    logger.info("Latency CSV saved to %s", csv_path)
    return summary
