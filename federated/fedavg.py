"""
Lightweight Federated Learning simulation using manual FedAvg.

This does NOT use TensorFlow Federated. Instead it:
    1. Partitions training data across N simulated clients.
    2. Each client trains a local copy of the model for a few epochs.
    3. The server collects the client model weights and averages them
       (Federated Averaging).
    4. The averaged weights are broadcast back to all clients.
    5. Repeat for multiple federated rounds.

Logs per-round metrics to federated_log.csv.
"""

from __future__ import annotations

import copy
import csv
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from numpy.typing import NDArray

from config.settings import Config

logger = logging.getLogger(__name__)


class FederatedTrainer:
    """FedAvg simulation over simulated local clients."""

    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.log: List[Dict[str, Any]] = []

    # ── data partitioning ────────────────────────────────────────────────

    @staticmethod
    def partition_data(
        X: NDArray, y: NDArray, n_clients: int
    ) -> List[Tuple[NDArray, NDArray]]:
        """Split data into *n_clients* roughly equal contiguous partitions."""
        indices = np.array_split(np.arange(len(X)), n_clients)
        partitions = [(X[idx], y[idx]) for idx in indices]
        for i, (xp, yp) in enumerate(partitions):
            logger.info("Client %d: %d samples", i, len(xp))
        return partitions

    # ── model builder ────────────────────────────────────────────────────

    def _build_model(self):
        """Build a fresh LEF-BiLSTM model for each client."""
        from models.lef_bilstm_model import LEFBiLSTMModel  # noqa: E402

        model = LEFBiLSTMModel(self.cfg)
        return model.model  # return the raw Keras model

    # ── FedAvg core ──────────────────────────────────────────────────────

    @staticmethod
    def _average_weights(
        all_weights: List[List[NDArray]],
    ) -> List[NDArray]:
        """Compute element-wise average of model weights (FedAvg)."""
        avg = []
        for layer_idx in range(len(all_weights[0])):
            layer_weights = np.array(
                [w[layer_idx] for w in all_weights]
            )
            avg.append(np.mean(layer_weights, axis=0))
        return avg

    # ── training loop ────────────────────────────────────────────────────

    def run(
        self,
        X_train: NDArray,
        y_train: NDArray,
        X_val: NDArray,
        y_val: NDArray,
    ) -> List[Dict[str, Any]]:
        """
        Execute the federated learning simulation.

        Returns the log list (also saved to CSV).
        """
        n_clients = self.cfg.NUM_CLIENTS
        n_rounds = self.cfg.NUM_FED_ROUNDS
        local_epochs = self.cfg.FED_LOCAL_EPOCHS

        partitions = self.partition_data(X_train, y_train, n_clients)

        # Initialise global model
        global_model = self._build_model()
        global_weights = global_model.get_weights()

        self.log = []

        for rnd in range(1, n_rounds + 1):
            logger.info("─── Federated Round %d / %d ───", rnd, n_rounds)

            client_weights = []
            client_losses = []

            for c_id in range(n_clients):
                # Create client model with current global weights
                client_model = self._build_model()
                client_model.set_weights(copy.deepcopy(global_weights))

                X_c, y_c = partitions[c_id]
                hist = client_model.fit(
                    X_c,
                    y_c,
                    epochs=local_epochs,
                    batch_size=self.cfg.BATCH_SIZE,
                    verbose=0,
                )
                c_loss = hist.history["loss"][-1]
                client_losses.append(c_loss)
                client_weights.append(client_model.get_weights())
                logger.info(
                    "  Client %d — local loss: %.6f", c_id, c_loss
                )

            # FedAvg
            global_weights = self._average_weights(client_weights)
            global_model.set_weights(global_weights)

            # Evaluate global model on validation set
            val_loss = global_model.evaluate(X_val, y_val, verbose=0)
            # model.evaluate returns [loss, mae]
            if isinstance(val_loss, list):
                val_loss_val = val_loss[0]
            else:
                val_loss_val = val_loss

            round_log = {
                "Round": rnd,
                "Avg_Client_Loss": round(float(np.mean(client_losses)), 6),
                "Global_Val_Loss": round(float(val_loss_val), 6),
            }
            # Add per-client losses
            for c_id, cl in enumerate(client_losses):
                round_log[f"Client_{c_id}_Loss"] = round(float(cl), 6)

            self.log.append(round_log)
            logger.info(
                "  Round %d — avg client loss: %.6f, global val loss: %.6f",
                rnd,
                round_log["Avg_Client_Loss"],
                round_log["Global_Val_Loss"],
            )

        # Save global model
        model_path = self.cfg.MODELS_DIR / "federated_global_model.keras"
        model_path.parent.mkdir(parents=True, exist_ok=True)
        global_model.save(str(model_path))
        logger.info("Federated global model saved to %s", model_path)

        # Save log
        self._save_log()

        return self.log

    def _save_log(self) -> None:
        """Write federated_log.csv."""
        if not self.log:
            return

        path = self.cfg.METRICS_DIR / "federated_log.csv"
        path.parent.mkdir(parents=True, exist_ok=True)

        fieldnames = list(self.log[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(self.log)

        logger.info("Federated log saved to %s", path)
