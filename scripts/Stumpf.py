from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

EPS = 1e-9
FEATURES = ["Band_1", "Band_2"]


def _finite_rows(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)


def stumpf_index(df: pd.DataFrame) -> np.ndarray:
    b1 = df["Band_1"].to_numpy(dtype=float)
    b2 = df["Band_2"].to_numpy(dtype=float)

    numerator = np.log(np.maximum(1000.0 * np.pi * b1, EPS))
    denominator = np.log(np.maximum(1000.0 * np.pi * b2, EPS))
    denominator = np.where(np.abs(denominator) < EPS, np.nan, denominator)

    return numerator / denominator


def fit_model(train_df: pd.DataFrame) -> Dict[str, Any]:
    mask = _finite_rows(train_df, FEATURES + ["depth"])
    subset = train_df.loc[mask, FEATURES + ["depth"]]

    index = stumpf_index(subset[FEATURES])
    depth = subset["depth"].to_numpy(dtype=float)

    valid = np.isfinite(index) & np.isfinite(depth)
    design = np.column_stack([np.ones(valid.sum()), index[valid]])
    coefficients = np.linalg.lstsq(design, depth[valid], rcond=None)[0]

    return {
        "kind": "stumpf",
        "features": FEATURES,
        "coefficients": coefficients,
    }


def predict(model: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    output = np.full(len(df), np.nan, dtype=float)
    mask = _finite_rows(df, FEATURES)

    if mask.any():
        index = stumpf_index(df.loc[mask, FEATURES])
        valid = np.isfinite(index)
        positions = np.where(mask)[0]
        coefficients = model["coefficients"]
        output[positions[valid]] = coefficients[0] + coefficients[1] * index[valid]

    return output
