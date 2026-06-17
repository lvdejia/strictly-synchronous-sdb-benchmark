from __future__ import annotations

import shutil
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

import Geo_LightGBM
import Geo_RF
import LightGBM
import LogLinear
import RandomForest
import Stumpf
from Metrics import calculate

ROOT = (
    Path(__file__).resolve().parents[1]
    if Path(__file__).resolve().parent.name == "scripts"
    else Path(__file__).resolve().parent
)
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "train2000_rest_geo6"

TRAIN_SOURCE = DATA_DIR / "xunlian_original.csv"
VALIDATION_SOURCE = DATA_DIR / "yanzheng_original.csv"
IMAGE_SOURCE = DATA_DIR / "out_image31.csv"

TARGET = "depth"
LONLAT = ["Longitude", "Latitude"]
BANDS = ["Band_1", "Band_2", "Band_3", "Band_4"]
POINT_COLUMNS = LONLAT + [TARGET] + BANDS
IMAGE_COLUMNS = LONLAT + BANDS

NODATA_VALUES = [32767, -32768, 9999, -9999]
SEED = 222
TRAIN_SAMPLES = 2000
TOTAL_SAMPLES = 3816
VALIDATION_SAMPLES = 1816

MODELS = [
    ("stumpf", "Stumpf", Stumpf),
    ("log_linear", "Log-linear", LogLinear),
    ("random_forest", "Random Forest", RandomForest),
    ("lightgbm", "LightGBM", LightGBM),
    ("geo_rf", "Geo_RF", Geo_RF),
    ("geo_lightgbm", "Geo_LightGBM", Geo_LightGBM),
]


def read_numeric(path: Path, columns: list[str], drop_invalid: bool) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Missing file: {path}")

    frame = pd.read_csv(path)
    missing = [column for column in columns if column not in frame.columns]
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    frame = frame[columns].copy()
    for column in columns:
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.replace(NODATA_VALUES, np.nan)

    if drop_invalid:
        mask = np.isfinite(frame[columns].to_numpy(dtype=float)).all(axis=1)
        frame = frame.loc[mask]

    return frame.reset_index(drop=True)


def make_split() -> tuple[pd.DataFrame, pd.DataFrame]:
    first = read_numeric(TRAIN_SOURCE, POINT_COLUMNS, drop_invalid=True)
    second = read_numeric(VALIDATION_SOURCE, POINT_COLUMNS, drop_invalid=True)

    merged = pd.concat([first, second], ignore_index=True)
    merged = merged.drop_duplicates(subset=POINT_COLUMNS).reset_index(drop=True)

    if len(merged) != TOTAL_SAMPLES:
        raise ValueError(
            f"Expected {TOTAL_SAMPLES} usable unique samples, found {len(merged)}."
        )

    generator = np.random.default_rng(SEED)
    indices = np.arange(len(merged))
    generator.shuffle(indices)

    train = merged.iloc[indices[:TRAIN_SAMPLES]].reset_index(drop=True)
    validation = merged.iloc[indices[TRAIN_SAMPLES:]].reset_index(drop=True)

    if len(train) != TRAIN_SAMPLES or len(validation) != VALIDATION_SAMPLES:
        raise RuntimeError("The 2000/1816 split was not produced.")

    overlap = train.merge(validation, on=POINT_COLUMNS, how="inner")
    if not overlap.empty:
        raise RuntimeError("Training and validation sets overlap.")

    return train, validation


def depth_bin_metrics(predictions: pd.DataFrame) -> pd.DataFrame:
    bins = [
        ("0-2", 0.0, 2.0),
        ("2-4", 2.0, 4.0),
        ("4-6", 4.0, 6.0),
        (">6", 6.0, np.inf),
    ]

    rows = []
    for label, lower, upper in bins:
        if np.isinf(upper):
            mask = predictions[TARGET] >= lower
        else:
            mask = (
                (predictions[TARGET] >= lower)
                & (predictions[TARGET] < upper)
            )

        row = {
            "depth_bin_m": label,
            "n": int(mask.sum()),
            "percentage": float(mask.mean() * 100.0),
        }

        truth = predictions.loc[mask, TARGET].to_numpy(dtype=float)
        for model_key, _, _ in MODELS:
            estimate = predictions.loc[mask, model_key].to_numpy(dtype=float)
            valid = np.isfinite(truth) & np.isfinite(estimate)
            row[f"{model_key}_RMSE"] = float(
                np.sqrt(np.mean((estimate[valid] - truth[valid]) ** 2))
            )

        rows.append(row)

    return pd.DataFrame(rows)


def main() -> None:
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    (OUTPUT_DIR / "models").mkdir(parents=True)
    (OUTPUT_DIR / "fullscene_final").mkdir(parents=True)

    train, validation = make_split()
    image = read_numeric(IMAGE_SOURCE, IMAGE_COLUMNS, drop_invalid=False)

    train.to_csv(
        OUTPUT_DIR / "train2000.csv",
        index=False,
        encoding="utf-8-sig",
    )
    validation.to_csv(
        OUTPUT_DIR / "validation_rest.csv",
        index=False,
        encoding="utf-8-sig",
    )

    validation_output = validation.copy()
    image_output = image[LONLAT].copy()

    metric_rows = []
    log_rows = []

    for model_key, display_name, module in MODELS:
        started = time.perf_counter()
        fitted = module.fit_model(train)
        elapsed = time.perf_counter() - started

        validation_prediction = module.predict(fitted, validation)
        image_prediction = module.predict(fitted, image)

        validation_output[model_key] = validation_prediction
        image_output[model_key] = image_prediction

        metrics = calculate(
            validation[TARGET].to_numpy(dtype=float),
            validation_prediction,
        )
        metrics.update(
            {
                "model_key": model_key,
                "model": display_name,
            }
        )
        metric_rows.append(metrics)

        log_rows.append(
            {
                "model_key": model_key,
                "model": display_name,
                "train_samples": len(train),
                "validation_samples": len(validation),
                "train_seconds": elapsed,
            }
        )

        joblib.dump(fitted, OUTPUT_DIR / "models" / f"{model_key}.joblib")

        fullscene = image[LONLAT].copy()
        fullscene[model_key] = image_prediction
        fullscene.to_csv(
            OUTPUT_DIR / "fullscene_final" / f"{model_key}_fullscene.csv",
            index=False,
            encoding="utf-8-sig",
        )

        print(
            f"{display_name}: "
            f"R2={metrics['R2']:.6f}, "
            f"RMSE={metrics['RMSE']:.6f}, "
            f"MAE={metrics['MAE']:.6f}"
        )

    validation_output.to_csv(
        OUTPUT_DIR / "validation_predictions_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    image_output.to_csv(
        OUTPUT_DIR / "image_predictions_final.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(metric_rows).to_csv(
        OUTPUT_DIR / "metrics_summary_oof.csv",
        index=False,
        encoding="utf-8-sig",
    )
    pd.DataFrame(log_rows).to_csv(
        OUTPUT_DIR / "training_log.csv",
        index=False,
        encoding="utf-8-sig",
    )
    depth_bin_metrics(validation_output).to_csv(
        OUTPUT_DIR / "depth_bin_metrics.csv",
        index=False,
        encoding="utf-8-sig",
    )

    pd.DataFrame(
        [
            {"item": "total_samples", "value": len(train) + len(validation)},
            {"item": "train_samples", "value": len(train)},
            {"item": "validation_samples", "value": len(validation)},
            {"item": "random_seed", "value": SEED},
        ]
    ).to_csv(
        OUTPUT_DIR / "data_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )


if __name__ == "__main__":
    main()
