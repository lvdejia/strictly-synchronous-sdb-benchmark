from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

FEATURES = ["Band_1", "Band_2", "Band_3", "Band_4"]
SEED = 222


def _finite_rows(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)


def fit_model(train_df: pd.DataFrame) -> Dict[str, Any]:
    mask = _finite_rows(train_df, FEATURES + ["depth"])
    model = RandomForestRegressor(
        n_estimators=100,
        random_state=SEED,
        n_jobs=-1,
    )
    model.fit(
        train_df.loc[mask, FEATURES],
        train_df.loc[mask, "depth"].to_numpy(dtype=float),
    )

    return {
        "kind": "random_forest",
        "features": FEATURES,
        "model": model,
    }


def predict(model: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    output = np.full(len(df), np.nan, dtype=float)
    mask = _finite_rows(df, FEATURES)

    if mask.any():
        output[mask] = model["model"].predict(df.loc[mask, FEATURES])

    return output
