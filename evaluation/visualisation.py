"""
Publication-quality figure generation.

Generates all figures required for the research paper / implementation evidence:
    - Prediction graph (actual vs predicted for each model)
    - Training loss curves
    - Validation loss curves
    - RMSE comparison bar chart
    - MAE comparison bar chart
    - R² comparison bar chart
    - Latency comparison bar chart
    - Federated convergence plot
    - Edge latency distribution
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np

from config.settings import Config

logger = logging.getLogger(__name__)

# ── Style ────────────────────────────────────────────────────────────────

COLORS = [
    "#2196F3",  # Blue
    "#FF9800",  # Orange
    "#4CAF50",  # Green
    "#E91E63",  # Pink
    "#9C27B0",  # Purple
]

plt.rcParams.update(
    {
        "figure.dpi": 150,
        "savefig.dpi": 300,
        "font.size": 11,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "figure.figsize": (10, 5),
        "axes.grid": True,
        "grid.alpha": 0.3,
    }
)


def _save(fig, path: Path, name: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    logger.info("Figure saved: %s", path)


# ── Individual generators ────────────────────────────────────────────────


def plot_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str,
    cfg: Config,
) -> None:
    """Plot actual vs predicted values for a single model."""
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_true, label="Actual", alpha=0.7, linewidth=0.8)
    ax.plot(y_pred, label=f"Predicted ({model_name})", alpha=0.7, linewidth=0.8)
    ax.set_title(f"Energy Consumption Prediction — {model_name}")
    ax.set_xlabel("Time Step")
    ax.set_ylabel("Global Active Power (kW)")
    ax.legend()
    fig.tight_layout()
    _save(fig, cfg.FIGURES_DIR / f"predictions_{model_name}.png", model_name)


def plot_training_curves(
    history: Dict[str, Any], model_name: str, cfg: Config
) -> None:
    """Plot training and validation loss curves."""
    if "loss" not in history:
        return

    fig, ax = plt.subplots()
    ax.plot(history["loss"], label="Train Loss", color=COLORS[0])
    if "val_loss" in history:
        ax.plot(history["val_loss"], label="Val Loss", color=COLORS[1])
    ax.set_title(f"Training Curves — {model_name}")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MSE Loss")
    ax.legend()
    fig.tight_layout()
    _save(fig, cfg.FIGURES_DIR / f"training_curves_{model_name}.png", model_name)


def plot_comparison_bar(
    results: Dict[str, Dict[str, float]],
    metric: str,
    ylabel: str,
    title: str,
    cfg: Config,
    filename: str,
) -> None:
    """Generic bar chart for comparing a metric across models."""
    models = list(results.keys())
    values = [results[m].get(metric, 0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(models, values, color=COLORS[: len(models)], edgecolor="white")

    # Annotate values
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.02,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax.set_ylabel(ylabel)
    ax.set_title(title)
    fig.tight_layout()
    _save(fig, cfg.FIGURES_DIR / filename, metric)


def plot_latency_comparison(
    latency_data: Dict[str, float], cfg: Config
) -> None:
    """Bar chart comparing Keras vs TFLite inference latency."""
    labels = ["Keras", "TFLite"]
    means = [latency_data["Keras_Mean_ms"], latency_data["TFLite_Mean_ms"]]
    stds = [latency_data["Keras_Std_ms"], latency_data["TFLite_Std_ms"]]

    fig, ax = plt.subplots(figsize=(6, 5))
    bars = ax.bar(labels, means, yerr=stds, color=[COLORS[0], COLORS[2]], capsize=5)

    for bar, val in zip(bars, means):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(means) * 0.05,
            f"{val:.3f} ms",
            ha="center",
            fontsize=10,
        )

    ax.set_ylabel("Inference Latency (ms)")
    ax.set_title(f"Edge Latency: Keras vs TFLite (Speedup: {latency_data['Speedup']}×)")
    fig.tight_layout()
    _save(fig, cfg.FIGURES_DIR / "latency_comparison.png", "latency")


def plot_federated_convergence(
    fed_log: List[Dict[str, Any]], cfg: Config
) -> None:
    """Plot federated learning convergence (global val loss per round)."""
    if not fed_log:
        return

    rounds = [entry["Round"] for entry in fed_log]
    val_losses = [entry["Global_Val_Loss"] for entry in fed_log]
    avg_client = [entry["Avg_Client_Loss"] for entry in fed_log]

    fig, ax = plt.subplots()
    ax.plot(rounds, val_losses, "o-", label="Global Val Loss", color=COLORS[0])
    ax.plot(rounds, avg_client, "s--", label="Avg Client Loss", color=COLORS[1])
    ax.set_xlabel("Federated Round")
    ax.set_ylabel("Loss (MSE)")
    ax.set_title("Federated Learning Convergence (FedAvg)")
    ax.legend()
    fig.tight_layout()
    _save(fig, cfg.FIGURES_DIR / "federated_convergence.png", "federated")


def plot_edge_latency_distribution(
    edge_results: List[Dict], cfg: Config
) -> None:
    """Histogram of per-inference latencies from edge gateway simulation."""
    if not edge_results:
        return

    latencies = [r["latency_ms"] for r in edge_results]

    fig, ax = plt.subplots()
    ax.hist(latencies, bins=30, color=COLORS[2], edgecolor="white", alpha=0.8)
    ax.axvline(np.mean(latencies), color=COLORS[3], linestyle="--", label=f"Mean: {np.mean(latencies):.3f} ms")
    ax.set_xlabel("Latency (ms)")
    ax.set_ylabel("Frequency")
    ax.set_title("Edge Gateway Inference Latency Distribution")
    ax.legend()
    fig.tight_layout()
    _save(fig, cfg.FIGURES_DIR / "edge_latency_distribution.png", "edge_latency")


# ── Aggregate generator ─────────────────────────────────────────────────


def generate_all_figures(
    comparison_results: Dict[str, Dict[str, float]],
    cfg: Config,
    latency_data: Optional[Dict[str, float]] = None,
    fed_log: Optional[List[Dict[str, Any]]] = None,
    edge_results: Optional[List[Dict]] = None,
) -> None:
    """Generate all comparison figures."""
    logger.info("Generating publication-quality figures …")

    # Comparison bar charts
    plot_comparison_bar(
        comparison_results, "MAE", "MAE", "MAE Comparison Across Models",
        cfg, "mae_comparison.png",
    )
    plot_comparison_bar(
        comparison_results, "RMSE", "RMSE", "RMSE Comparison Across Models",
        cfg, "rmse_comparison.png",
    )
    plot_comparison_bar(
        comparison_results, "R2", "R²", "R² Comparison Across Models",
        cfg, "r2_comparison.png",
    )
    if any("Inference_Time_ms" in v for v in comparison_results.values()):
        plot_comparison_bar(
            comparison_results, "Inference_Time_ms", "Inference Time (ms)",
            "Inference Time Comparison", cfg, "inference_time_comparison.png",
        )

    # Latency
    if latency_data:
        plot_latency_comparison(latency_data, cfg)

    # Federated
    if fed_log:
        plot_federated_convergence(fed_log, cfg)

    # Edge
    if edge_results:
        plot_edge_latency_distribution(edge_results, cfg)

    logger.info("All figures saved to %s", cfg.FIGURES_DIR)
