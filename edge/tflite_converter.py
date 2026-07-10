"""
TensorFlow Lite conversion and inference utilities.

Converts a trained Keras model to TFLite format for edge deployment
and provides a helper to run inference using the TFLite Interpreter.
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Tuple

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


def convert_to_tflite(
    keras_model_path: str, tflite_output_path: str
) -> str:
    """
    Convert a saved Keras model to TensorFlow Lite format.

    Returns the path to the generated .tflite file.
    """
    import tensorflow as tf  # noqa: E402

    Path(tflite_output_path).parent.mkdir(parents=True, exist_ok=True)

    model = tf.keras.models.load_model(keras_model_path)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()

    with open(tflite_output_path, "wb") as f:
        f.write(tflite_model)

    size_kb = len(tflite_model) / 1024
    logger.info(
        "TFLite model saved to %s (%.1f KB)", tflite_output_path, size_kb
    )
    return tflite_output_path


def run_tflite_inference(
    tflite_path: str, X: NDArray
) -> Tuple[NDArray, float]:
    """
    Run inference using the TFLite Interpreter.

    Returns (predictions, total_inference_time_ms).
    """
    import tensorflow as tf  # noqa: E402

    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    predictions = []
    start = time.perf_counter()

    for i in range(len(X)):
        sample = X[i : i + 1].astype(np.float32)
        interpreter.set_tensor(input_details[0]["index"], sample)
        interpreter.invoke()
        pred = interpreter.get_tensor(output_details[0]["index"])
        predictions.append(pred[0, 0])

    total_time = (time.perf_counter() - start) * 1000  # ms
    logger.info(
        "TFLite inference on %d samples: %.2f ms (avg %.4f ms/sample)",
        len(X),
        total_time,
        total_time / max(len(X), 1),
    )
    return np.array(predictions).reshape(-1, 1), total_time
