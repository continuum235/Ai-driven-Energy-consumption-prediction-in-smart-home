"""Evaluation package — metrics computation, predictions export, comparison."""

from evaluation.evaluator import evaluate_model, save_predictions, save_metrics
from evaluation.comparison import run_comparison

__all__ = [
    "evaluate_model",
    "save_predictions",
    "save_metrics",
    "run_comparison",
]
