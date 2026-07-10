"""
MQTT simulation for IoT sensor data publishing.

Simulates a smart-home sensor that publishes energy readings to an MQTT
topic. A subscriber (the edge gateway) processes these readings for
inference.

IMPORTANT: This is a *simulation* — it does NOT require a running MQTT
broker. It uses paho-mqtt's internal publish/subscribe loop to demonstrate
the protocol flow, falling back to direct function calls if the broker
is unreachable.
"""

from __future__ import annotations

import json
import logging
import time
import threading
from typing import List, Optional

import numpy as np
from numpy.typing import NDArray

logger = logging.getLogger(__name__)


class MQTTSimulator:
    """
    Simulates MQTT publish/subscribe for IoT energy sensor data.

    When a broker is unavailable (common in dev/test), falls back to an
    in-process callback-based simulation that demonstrates the same
    data flow without requiring infrastructure.
    """

    def __init__(
        self,
        broker: str = "localhost",
        port: int = 1883,
        topic: str = "smart_home/energy",
    ) -> None:
        self.broker = broker
        self.port = port
        self.topic = topic
        self.received_messages: List[dict] = []
        self._use_fallback = False

    def _try_connect(self):
        """Attempt to connect to the MQTT broker."""
        try:
            import paho.mqtt.client as mqtt  # noqa: E402

            client = mqtt.Client(
                callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
                client_id="energy_sensor_sim",
            )
            client.connect(self.broker, self.port, keepalive=5)
            client.disconnect()
            return True
        except Exception as e:
            logger.warning(
                "MQTT broker not reachable at %s:%d (%s). "
                "Using in-process simulation fallback.",
                self.broker, self.port, e,
            )
            return False

    def simulate(
        self, data: NDArray, max_messages: int = 100, delay_s: float = 0.0
    ) -> List[dict]:
        """
        Publish sensor readings. If no broker is available, simulate
        the publish/subscribe loop in-process.
        """
        readings = data.flatten()[:max_messages]
        logger.info(
            "MQTT simulation: publishing %d readings to topic '%s'",
            len(readings), self.topic,
        )

        if self._try_connect():
            return self._run_with_broker(readings, delay_s)
        else:
            return self._run_fallback(readings, delay_s)

    def _run_with_broker(
        self, readings: NDArray, delay_s: float
    ) -> List[dict]:
        """Use a real MQTT broker."""
        import paho.mqtt.client as mqtt  # noqa: E402

        results = []

        # Subscriber
        def on_message(client, userdata, msg):
            payload = json.loads(msg.payload.decode())
            results.append(payload)

        sub = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="edge_gateway_sub",
        )
        sub.on_message = on_message
        sub.connect(self.broker, self.port)
        sub.subscribe(self.topic)
        sub.loop_start()

        # Publisher
        pub = mqtt.Client(
            callback_api_version=mqtt.CallbackAPIVersion.VERSION2,
            client_id="sensor_pub",
        )
        pub.connect(self.broker, self.port)

        for i, val in enumerate(readings):
            payload = {
                "sensor_id": "household_power_01",
                "timestamp": time.time(),
                "reading_index": int(i),
                "global_active_power": round(float(val), 6),
            }
            pub.publish(self.topic, json.dumps(payload))
            if delay_s > 0:
                time.sleep(delay_s)

        time.sleep(1)  # drain remaining messages
        sub.loop_stop()
        sub.disconnect()
        pub.disconnect()

        self.received_messages = results
        logger.info("MQTT broker simulation complete: %d messages", len(results))
        return results

    def _run_fallback(
        self, readings: NDArray, delay_s: float
    ) -> List[dict]:
        """In-process simulation (no broker required)."""
        results = []

        for i, val in enumerate(readings):
            payload = {
                "sensor_id": "household_power_01",
                "timestamp": time.time(),
                "reading_index": int(i),
                "global_active_power": round(float(val), 6),
            }
            # Simulate publish → subscribe
            results.append(payload)
            if delay_s > 0:
                time.sleep(delay_s)

        self.received_messages = results
        logger.info(
            "In-process MQTT simulation complete: %d messages", len(results)
        )
        return results
