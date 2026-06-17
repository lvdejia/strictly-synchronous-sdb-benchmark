from __future__ import annotations

from typing import Any, Dict

import numpy as np
import pandas as pd

FEATURES = ["Band_1", "Band_2", "Band_3", "Band_4"]
SEED = 222


def _finite_rows(df: pd.DataFrame, columns: list[str]) -> np.ndarray:
    return np.isfinite(df[columns].to_numpy(dtype=float)).all(axis=1)


def fit_model(train_df: pd.DataFrame) -> Dict[str, Any]:
    try:
        import lightgbm as lgb
    except ImportError as exc:
        raise ImportError("Install LightGBM with: pip install lightgbm") from exc

    mask = _finite_rows(train_df, FEATURES + ["depth"])
    model = lgb.LGBMRegressor(
        objective="regression",
        boosting_type="gbdt",
        n_estimators=2000,
        learning_rate=0.03,
        num_leaves=31,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    model.fit(
        train_df.loc[mask, FEATURES],
        train_df.loc[mask, "depth"].to_numpy(dtype=float),
    )

    return {
        "kind": "lightgbm",
        "features": FEATURES,
        "model": model,
    }


def predict(model: Dict[str, Any], df: pd.DataFrame) -> np.ndarray:
    output = np.full(len(df), np.nan, dtype=float)
    mask = _finite_rows(df, FEATURES)

    if mask.any():
        output[mask] = model["model"].predict(df.loc[mask, FEATURES])

    return output
