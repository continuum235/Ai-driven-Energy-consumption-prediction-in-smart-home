"""
Simulated IoT edge gateway for smart-home energy prediction.

The gateway:
    1. Receives sensor readings (simulated or via MQTT).
    2. Runs on-device inference using a TFLite model.
    3. Logs results and latency.

This module can operate standalone or be driven by the MQTT simulator.
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class EdgeGateway:
    """Simulated IoT edge gateway performing local inference."""

    def __init__(
        self,
        tflite_path: str,
        time_step: int = 30,
    ) -> None:
        import tensorflow as tf  # noqa: E402

        self.time_step = time_step
        self.tflite_path = tflite_path

        # Load interpreter
        self.interpreter = tf.lite.Interpreter(model_path=tflite_path)
        self.interpreter.allocate_tensors()
        self.input_details = self.interpreter.get_input_details()
        self.output_details = self.interpreter.get_output_details()

        # Sliding buffer for streaming inference
        self.buffer: List[float] = []
        self.results: List[Dict] = []

        logger.info(
            "EdgeGateway initialised with model: %s (window=%d)",
            tflite_path,
            time_step,
        )

    def ingest(self, reading: float) -> Optional[Dict]:
        """
        Ingest a single sensor reading.

        When the buffer reaches *time_step* readings, run inference and
        return the result dict. Otherwise return None.
        """
        self.buffer.append(reading)

        if len(self.buffer) < self.time_step:
            return None

        # Prepare input
        window = np.array(
            self.buffer[-self.time_step :], dtype=np.float32
        ).reshape(1, self.time_step, 1)

        # Run inference
        start = time.perf_counter()
        self.interpreter.set_tensor(self.input_details[0]["index"], window)
        self.interpreter.invoke()
        prediction = self.interpreter.get_tensor(
            self.output_details[0]["index"]
        )[0, 0]
        latency_ms = (time.perf_counter() - start) * 1000

        result = {
            "prediction": float(prediction),
            "latency_ms": round(latency_ms, 4),
            "buffer_size": len(self.buffer),
        }
        self.results.append(result)
        return result

    def simulate_stream(self, data: NDArray) -> List[Dict]:
        """Feed an array of readings through the gateway sequentially."""
        logger.info(
            "Simulating edge stream with %d readings …", len(data)
        )
        self.buffer = []
        self.results = []

        for val in data.flatten():
            self.ingest(float(val))

        logger.info(
            "Edge simulation complete — %d inferences performed.",
            len(self.results),
        )
        return self.results
