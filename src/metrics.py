"""Evaluation metrics. MAPE is the primary metric; WAPE, MAE and RMSE are reported for interpretation."""

import numpy as np


def mape(y_true, y_pred) -> float:
    """Mean Absolute Percentage Error, in percent. Every load has equal weight."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred) / np.abs(y_true)) * 100)


def wape(y_true, y_pred) -> float:
    """Weighted Absolute Percentage Error, in percent: total absolute error over total actual."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sum(np.abs(y_true - y_pred)) / np.sum(np.abs(y_true)) * 100)


def mae(y_true, y_pred) -> float:
    """Mean Absolute Error, in dollars."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.mean(np.abs(y_true - y_pred)))


def rmse(y_true, y_pred) -> float:
    """Root Mean Squared Error, in dollars."""
    y_true, y_pred = np.asarray(y_true, float), np.asarray(y_pred, float)
    return float(np.sqrt(np.mean((y_true - y_pred) ** 2)))


def evaluate(y_true, y_pred) -> dict[str, float]:
    """All metrics at once, primary metric first."""
    return {
        "mape": mape(y_true, y_pred),
        "wape": wape(y_true, y_pred),
        "mae": mae(y_true, y_pred),
        "rmse": rmse(y_true, y_pred),
    }
