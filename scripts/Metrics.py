from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def calculate(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    truth = np.asarray(y_true, dtype=float)
    prediction = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(truth) & np.isfinite(prediction)
    truth = truth[mask]
    prediction = prediction[mask]

    residual = prediction - truth
    absolute_error = np.abs(residual)

    return {
        "n": int(len(truth)),
        "R2": float(r2_score(truth, prediction)),
        "RMSE": float(np.sqrt(mean_squared_error(truth, prediction))),
        "MAE": float(mean_absolute_error(truth, prediction)),
        "Bias": float(np.mean(residual)),
        "AE_le_0.5_pct": float(np.mean(absolute_error <= 0.5) * 100.0),
        "AE_le_1.0_pct": float(np.mean(absolute_error <= 1.0) * 100.0),
    }
