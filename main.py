"""
main.py — Entry point for the AI-Driven Energy Consumption Prediction pipeline.

Orchestrates:
    1. Data preprocessing
    2. Model training and evaluation (all 5 architectures)
    3. Baseline comparison
    4. Edge computing pipeline (TFLite conversion + latency measurement)
    5. MQTT IoT simulation
    6. Federated learning simulation (FedAvg)
    7. Publication-quality figure generation
    8. Report and metrics export

Usage:
    python main.py
"""

from __future__ import annotations

import csv
import logging
import os
import sys
import time
from pathlib import Path

import numpy as np

# ── Ensure project root is on sys.path ───────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from config.settings import Config

# ── Logging setup ────────────────────────────────────────────────────────

def setup_logging(cfg: Config) -> None:
    """Configure logging to both console and file."""
    cfg.ensure_dirs()
    log_file = cfg.LOGS_DIR / "training.log"

    handlers = [
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(str(log_file), mode="w"),
    ]

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )


# ── Main pipeline ────────────────────────────────────────────────────────

def main() -> None:
    cfg = Config()
    setup_logging(cfg)
    logger = logging.getLogger(__name__)

    logger.info("=" * 70)
    logger.info("  AI-Driven Energy Consumption Prediction Pipeline")
    logger.info("=" * 70)

    # ── 0. Seed ──────────────────────────────────────────────────────
    import tensorflow as tf
    np.random.seed(cfg.SEED)
    tf.random.set_seed(cfg.SEED)

    # ── 1. Check dataset location ────────────────────────────────────
    # Support both dataset/ subfolder and project root for the data file
    if not cfg.DATA_PATH.exists():
        alt_path = cfg.BASE_DIR / cfg.DATA_FILENAME
        if alt_path.exists():
            logger.info("Moving data.txt to dataset/ directory …")
            cfg.DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
            import shutil
            shutil.copy2(str(alt_path), str(cfg.DATA_PATH))
        else:
            logger.error(
                "Dataset not found at %s or %s. "
                "Please download from UCI and place data.txt in the dataset/ directory.",
                cfg.DATA_PATH, alt_path,
            )
            sys.exit(1)

    # ── 2. Preprocessing ─────────────────────────────────────────────
    from preprocessing.pipeline import load_and_preprocess

    logger.info("─── STAGE 1: Preprocessing ───")
    data = load_and_preprocess(cfg)
    logger.info(
        "Preprocessing complete. Train=%s Val=%s Test=%s",
        data.X_train.shape, data.X_val.shape, data.X_test.shape,
    )

    # ── 3. Baseline Comparison (Train + Evaluate all models) ─────────
    from evaluation.comparison import run_comparison
    from evaluation.visualisation import (
        plot_predictions,
        plot_training_curves,
        generate_all_figures,
    )

    logger.info("─── STAGE 2: Baseline Comparison ───")
    comparison_results = run_comparison(
        data.X_train, data.y_train,
        data.X_val, data.y_val,
        data.X_test, data.y_test,
        data.scaler, cfg,
    )

    # Generate per-model prediction plots
    from models.registry import build_model

    for model_name in cfg.MODEL_NAMES:
        model = build_model(model_name, cfg)
        # Load the trained model
        try:
            model_path = str(cfg.MODELS_DIR / f"{model_name}_final.keras")
            model.load(model_path)
        except Exception:
            try:
                model_path = str(cfg.MODELS_DIR / f"{model_name}_final.pkl")
                model.load(model_path)
            except Exception:
                logger.warning("Could not load %s for plotting, skipping.", model_name)
                continue

        y_pred_scaled = model.predict(data.X_test)
        y_pred = data.scaler.inverse_transform(y_pred_scaled.reshape(-1, 1)).flatten()
        y_true = data.scaler.inverse_transform(data.y_test.reshape(-1, 1)).flatten()
        plot_predictions(y_true, y_pred, model_name, cfg)

        # Plot training curves if available
        hist_path = cfg.METRICS_DIR / f"{model_name}_training_history.csv"
        if hist_path.exists():
            import pandas as pd
            hist_df = pd.read_csv(hist_path)
            history = {
                "loss": hist_df["Loss"].tolist(),
                "val_loss": hist_df["Val_Loss"].tolist() if "Val_Loss" in hist_df.columns else [],
            }
            plot_training_curves(history, model_name, cfg)

    # ── 4. Save Combined Metrics ─────────────────────────────────────
    from evaluation.evaluator import save_metrics

    save_metrics(comparison_results, cfg.METRICS_DIR / "metrics.csv")

    # ── 5. Edge Computing Pipeline ───────────────────────────────────
    logger.info("─── STAGE 3: Edge Computing ───")

    from edge.tflite_converter import convert_to_tflite
    from edge.latency import measure_latency
    from edge.edge_gateway import EdgeGateway
    from edge.mqtt_simulator import MQTTSimulator

    # Use LEF-BiLSTM for edge deployment
    lef_model_path = str(cfg.MODELS_DIR / "LEF-BiLSTM_final.keras")
    tflite_path = str(cfg.TFLITE_DIR / "model.tflite")

    try:
        convert_to_tflite(lef_model_path, tflite_path)

        # Measure latency
        latency_data = measure_latency(
            lef_model_path, tflite_path, data.X_test, cfg
        )

        # Edge gateway simulation
        logger.info("Running edge gateway simulation …")
        gateway = EdgeGateway(tflite_path, cfg.TIME_STEP)
        test_raw = data.scaler.inverse_transform(
            data.y_test.reshape(-1, 1)
        ).flatten()
        edge_results = gateway.simulate_stream(test_raw[:200])

        # Save edge results
        edge_csv = cfg.METRICS_DIR / "edge_inference_log.csv"
        with open(edge_csv, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["prediction", "latency_ms", "buffer_size"])
            writer.writeheader()
            writer.writerows(edge_results)
        logger.info("Edge inference log saved to %s", edge_csv)

        # MQTT simulation
        logger.info("Running MQTT simulation …")
        mqtt_sim = MQTTSimulator(cfg.MQTT_BROKER, cfg.MQTT_PORT, cfg.MQTT_TOPIC)
        mqtt_messages = mqtt_sim.simulate(test_raw[:50])
        logger.info("MQTT simulation complete: %d messages", len(mqtt_messages))

    except Exception as e:
        logger.error("Edge computing stage failed: %s", e, exc_info=True)
        latency_data = None
        edge_results = None

    # ── 6. Federated Learning ────────────────────────────────────────
    logger.info("─── STAGE 4: Federated Learning ───")

    from federated.fedavg import FederatedTrainer

    try:
        fed_trainer = FederatedTrainer(cfg)
        fed_log = fed_trainer.run(
            data.X_train, data.y_train,
            data.X_val, data.y_val,
        )
    except Exception as e:
        logger.error("Federated learning stage failed: %s", e, exc_info=True)
        fed_log = None

    # ── 7. Visualisations ────────────────────────────────────────────
    logger.info("─── STAGE 5: Figure Generation ───")

    generate_all_figures(
        comparison_results, cfg,
        latency_data=latency_data,
        fed_log=fed_log,
        edge_results=edge_results,
    )

    # ── 8. Save Hyperparameters ──────────────────────────────────────
    logger.info("─── STAGE 6: Export Artefacts ───")

    hp_path = cfg.METRICS_DIR / "hyperparameters.txt"
    with open(hp_path, "w", encoding="utf-8") as f:
        f.write("Hyperparameters\n")
        f.write("=" * 40 + "\n")
        for field_name in [
            "SEED", "NUM_ROWS", "TIME_STEP", "TRAIN_RATIO", "VAL_RATIO",
            "EPOCHS", "BATCH_SIZE", "LEARNING_RATE",
            "EARLY_STOPPING_PATIENCE", "REDUCE_LR_PATIENCE", "REDUCE_LR_FACTOR",
            "MIN_LR", "LSTM_UNITS", "DENSE_UNITS", "DROPOUT_RATE",
            "CNN_FILTERS", "CNN_KERNEL_SIZE", "ARIMA_ORDER", "ARIMA_MAX_SAMPLES",
            "NUM_CLIENTS", "NUM_FED_ROUNDS", "FED_LOCAL_EPOCHS",
            "EDGE_NUM_INFERENCE_RUNS",
        ]:
            f.write(f"{field_name}: {getattr(cfg, field_name)}\n")
    logger.info("Hyperparameters saved to %s", hp_path)

    # ── 9. Summary ───────────────────────────────────────────────────
    logger.info("=" * 70)
    logger.info("  PIPELINE COMPLETE")
    logger.info("=" * 70)
    logger.info("Outputs directory: %s", cfg.OUTPUTS_DIR)
    logger.info("Figures directory: %s", cfg.FIGURES_DIR)
    logger.info("")
    logger.info("Generated files:")
    for d in [cfg.METRICS_DIR, cfg.FIGURES_DIR, cfg.TFLITE_DIR, cfg.MODELS_DIR]:
        if d.exists():
            for f in sorted(d.iterdir()):
                logger.info("  %s", f.relative_to(cfg.BASE_DIR))


if __name__ == "__main__":
    main()