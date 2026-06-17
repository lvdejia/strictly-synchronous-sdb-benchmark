from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

EPS = 1e-9
FEATURES = ["Band_1", "Band_2", "Band_3", "Band_4"]


def _finite_rows(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)


def fit_model(train_df: pd.DataFrame) -> Dict[str, Any]:
    mask = _finite_rows(train_df, FEATURES + ["depth"])
    bands = train_df.loc[mask, FEATURES].to_numpy(dtype=float)
    depth = train_df.loc[mask, "depth"].to_numpy(dtype=float)

    positive = (bands > 0).all(axis=1)
    transformed = np.log(bands[positive] + EPS)

    regressor = LinearRegression(fit_intercept=True)
    regressor.fit(transformed, depth[positive])

    return {
        "kind": "log_linear",
        "features": FEATURES,
        "epsilon": EPS,
        "model": regressor,
    }


def predict(model: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    output = np.full(len(df), np.nan, dtype=float)
    finite = _finite_rows(df, FEATURES)

    positive = np.zeros(len(df), dtype=bool)
    positive[finite] = (
        df.loc[finite, FEATURES].to_numpy(dtype=float) > 0
    ).all(axis=1)

    valid = finite & positive
    if valid.any():
        transformed = np.log(
            df.loc[valid, FEATURES].to_numpy(dtype=float) + model["epsilon"]
        )
        output[valid] = model["model"].predict(transformed)

    return output
