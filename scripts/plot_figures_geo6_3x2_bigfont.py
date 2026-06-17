# -*- coding: utf-8 -*-
"""
plot_figures_geo6_3x2.py

最终大字体版：
1. 绘制 6 模型散点密度图；
2. 绘制 6 模型残差图；
3. 绘制 6 模型全景水深图；
4. 统一 Times New Roman；
5. 字号按论文图件可读性放大；
6. 输出 PNG + PDF + SVG。

运行：
    python scripts\plot_figures_geo6_3x2.py
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
from matplotlib.cm import ScalarMappable
from scipy.stats import gaussian_kde
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


# =========================
# 1. 基础配置
# =========================

ROOT = Path(__file__).resolve().parents[1] if Path(__file__).resolve().parent.name == "scripts" else Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "outputs" / "train2000_rest_geo6"
FIGURE_DIR = ROOT / "figures" / "train2000_rest_geo6" / "geo6_3x2"
IMAGE_FILE = DATA_DIR / "out_image31.csv"

MODEL_ORDER = [
    "stumpf",
    "log_linear",
    "random_forest",
    "lightgbm",
    "geo_rf",
    "geo_lightgbm",
]

MODEL_TITLES = {
    "stumpf": "Stumpf Model",
    "log_linear": "Log-linear Model",
    "random_forest": "Random Forest Model",
    "lightgbm": "LightGBM Model",
    "geo_rf": "Geo_RF Model",
    "geo_lightgbm": "Geo_LightGBM Model",
}

MODEL_YLABELS = {
    "stumpf": "SDB Stumpf (m)",
    "log_linear": "SDB Log-linear (m)",
    "random_forest": "SDB Random Forest (m)",
    "lightgbm": "SDB LightGBM (m)",
    "geo_rf": "SDB Geo_RF (m)",
    "geo_lightgbm": "SDB Geo_LightGBM (m)",
}

DENSITY_MIN = 0.0
DENSITY_MAX = 0.5
DEPTH_MIN = 0.0
DEPTH_MAX = 6.0
AXIS_MAX = 10.0
NDWI_THRESHOLD = -0.1
NODATA_VALUES = [32767, -32768, 9999, -9999]
DPI = 600


# =========================
# 2. 字体与图件样式
# =========================

TITLE_SIZE = 13
AXIS_LABEL_SIZE = 11
TICK_SIZE = 10
COLORBAR_LABEL_SIZE = 10
COLORBAR_TICK_SIZE = 9
PANEL_LABEL_SIZE = 11
METRIC_TEXT_SIZE = 10
LEGEND_SIZE = 10

plt.rcParams["font.family"] = "Times New Roman"
plt.rcParams["font.serif"] = ["Times New Roman"]
plt.rcParams["axes.unicode_minus"] = False
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["ps.fonttype"] = 42
# 最终稿：SVG 文字转路径，避免投稿/跨电脑字体丢失
plt.rcParams["svg.fonttype"] = "path"


# =========================
# 3. 通用函数
# =========================

def check_columns(df: pd.DataFrame, cols: List[str], name: str) -> None:
    """检查 DataFrame 是否包含必要列。"""
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{name} 缺少列：{missing}")


def to_numeric_df(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    """把指定列转成数值。"""
    out = df.copy()
    for col in cols:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def calc_metrics(y_true, y_pred) -> dict:
    """计算 R²、MAE、RMSE。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = np.isfinite(y_true) & np.isfinite(y_pred)
    y = y_true[mask]
    p = y_pred[mask]

    if len(y) == 0:
        return {"R2": np.nan, "MAE": np.nan, "RMSE": np.nan}

    return {
        "R2": r2_score(y, p),
        "MAE": mean_absolute_error(y, p),
        "RMSE": np.sqrt(mean_squared_error(y, p)),
    }


def density_to_range(x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """
    计算 KDE 点密度，并映射到统一 Density 范围。
    """
    try:
        xy = np.vstack([x, y])
        z = gaussian_kde(xy)(xy)
    except Exception:
        z = np.ones_like(x, dtype=float)

    z = np.asarray(z, dtype=float)

    if np.nanmax(z) > np.nanmin(z):
        z = (z - np.nanmin(z)) / (np.nanmax(z) - np.nanmin(z))
    else:
        z = np.zeros_like(z)

    return DENSITY_MIN + z * (DENSITY_MAX - DENSITY_MIN)


def add_colorbar(fig, ax, cmap, norm, label: str, ticks) -> None:
    """添加统一样式 colorbar。"""
    sm = ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])

    cbar = fig.colorbar(
        sm,
        ax=ax,
        orientation="vertical",
        fraction=0.045,
        pad=0.016,
    )
    cbar.set_label(label, fontsize=COLORBAR_LABEL_SIZE)
    cbar.set_ticks(ticks)
    cbar.ax.tick_params(
        labelsize=COLORBAR_TICK_SIZE,
        direction="out",
        length=2.5,
        width=0.8,
    )


def format_lon(x: float) -> str:
    return f"{x:.3f}°E"


def format_lat(y: float) -> str:
    return f"{y:.3f}°N"


def interior_ticks(vmin: float, vmax: float, n: int = 3) -> np.ndarray:
    """
    只取内部刻度，不取边界点，避免角点标签互相压住。
    """
    if n <= 0:
        return np.array([])
    return np.linspace(vmin, vmax, n + 2)[1:-1]


# =========================
# 4. 水体掩膜
# =========================

def water_mask(image_df: pd.DataFrame) -> np.ndarray:
    """基于 NDWI 生成水体掩膜。"""
    cols = ["Longitude", "Latitude", "Band_1", "Band_2", "Band_3", "Band_4"]
    check_columns(image_df, cols, "out_image31.csv")

    image_df = image_df[cols].copy()
    image_df = to_numeric_df(image_df, cols)
    image_df = image_df.replace(NODATA_VALUES, np.nan)

    green = image_df["Band_2"].to_numpy(dtype=float)
    nir = image_df["Band_4"].to_numpy(dtype=float)

    ndwi = (green - nir) / np.maximum(green + nir, 1e-9)

    mask = np.isfinite(ndwi) & (ndwi > NDWI_THRESHOLD)
    mask &= np.isfinite(image_df[cols].to_numpy(dtype=float)).all(axis=1)

    return mask


# =========================
# 5. 网格转换
# =========================

def pivot_map(df: pd.DataFrame, value_col: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """把经纬度点表转换为二维网格。"""
    table = df.pivot_table(
        index="Latitude",
        columns="Longitude",
        values=value_col,
        aggfunc="first"
    )

    table = table.sort_index(ascending=True)
    table = table.reindex(sorted(table.columns), axis=1)

    lon = table.columns.to_numpy(dtype=float)
    lat = table.index.to_numpy(dtype=float)
    z = table.to_numpy(dtype=float)

    return lon, lat, z


# =========================
# 6. 散点密度图
# =========================

def plot_scatter() -> None:
    """绘制 6 模型 3×2 散点密度图。"""
    val_file = OUTPUT_DIR / "validation_predictions_oof.csv"
    if not val_file.exists():
        raise FileNotFoundError(f"找不到：{val_file}")

    df = pd.read_csv(val_file)
    check_columns(df, ["depth"] + MODEL_ORDER, "validation_predictions_oof.csv")

    fig, axes = plt.subplots(3, 2, figsize=(10.8, 14.6))
    axes = axes.ravel()

    cmap = "Spectral_r"
    norm = mcolors.Normalize(vmin=DENSITY_MIN, vmax=DENSITY_MAX)
    ticks = np.linspace(DENSITY_MIN, DENSITY_MAX, 6)

    for i, model_key in enumerate(MODEL_ORDER):
        ax = axes[i]

        sub = df[["depth", model_key]].copy()
        sub = to_numeric_df(sub, ["depth", model_key]).dropna()

        x = sub["depth"].to_numpy(dtype=float)
        y = sub[model_key].to_numpy(dtype=float)

        z = density_to_range(x, y)
        order = np.argsort(z)
        x = x[order]
        y = y[order]
        z = z[order]

        ax.scatter(
            x,
            y,
            c=z,
            cmap=cmap,
            norm=norm,
            s=7,
            edgecolors="none"
        )

        ax.plot([0, AXIS_MAX], [0, AXIS_MAX], "k--", linewidth=0.9)

        if len(x) > 1:
            coef = np.polyfit(x, y, 1)
            xx = np.array([0, AXIS_MAX])
            yy = coef[0] * xx + coef[1]
            ax.plot(xx, yy, color="red", linewidth=1.1)

        m = calc_metrics(x, y)

        ax.text(
            0.04, 0.96,
            f"R²={m['R2']:.2f}\n\n"
            f"MAE={m['MAE']:.2f} m\n\n"
            f"RMSE={m['RMSE']:.2f} m",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=METRIC_TEXT_SIZE,
        )

        ax.text(
            0.98, 0.03,
            f"({chr(97+i)})",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold",
        )

        ax.set_title(MODEL_TITLES[model_key], fontsize=TITLE_SIZE, fontweight="bold", pad=6)
        ax.set_xlabel("In-situ depth (m)", fontsize=AXIS_LABEL_SIZE)
        ax.set_ylabel(MODEL_YLABELS[model_key], fontsize=AXIS_LABEL_SIZE)

        ax.set_xlim(0, AXIS_MAX)
        ax.set_ylim(0, AXIS_MAX)

        ax.set_xticks(np.arange(0, AXIS_MAX + 0.1, 3))
        ax.set_yticks(np.arange(0, AXIS_MAX + 0.1, 3))
        ax.tick_params(
            labelsize=TICK_SIZE,
            direction="out",
            length=3,
            width=0.9,
        )

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        add_colorbar(fig, ax, cmap, norm, "Point Density (dl)", ticks)

    fig.subplots_adjust(
        left=0.075,
        right=0.965,
        bottom=0.055,
        top=0.965,
        wspace=0.34,
        hspace=0.42,
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURE_DIR / "scatter_geo6_3x2_final_bigfont.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "scatter_geo6_3x2_final_bigfont.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "scatter_geo6_3x2_final_bigfont.svg", bbox_inches="tight")

    plt.close(fig)
    print(f"已保存散点图：{FIGURE_DIR / 'scatter_geo6_3x2_final_bigfont.png'}")


# =========================
# 7. 残差图
# =========================

def plot_residual() -> None:
    """绘制 6 模型残差直方图。"""
    val_file = OUTPUT_DIR / "validation_predictions_oof.csv"
    if not val_file.exists():
        raise FileNotFoundError(f"找不到：{val_file}")

    df = pd.read_csv(val_file)
    check_columns(df, ["depth"] + MODEL_ORDER, "validation_predictions_oof.csv")

    fig, ax = plt.subplots(figsize=(9.8, 6.4))
    bins = np.linspace(-5, 5, 90)

    for model_key in MODEL_ORDER:
        sub = df[["depth", model_key]].copy()
        sub = to_numeric_df(sub, ["depth", model_key]).dropna()

        residual = sub[model_key].to_numpy(dtype=float) - sub["depth"].to_numpy(dtype=float)

        ax.hist(
            residual,
            bins=bins,
            alpha=0.45,
            linewidth=0.6,
            edgecolor="gray",
            label=MODEL_TITLES[model_key].replace(" Model", "")
        )

    ax.axvline(0, color="black", linestyle="--", linewidth=1.1)

    ax.set_xlabel("Residual error: predicted - in-situ depth (m)", fontsize=AXIS_LABEL_SIZE)
    ax.set_ylabel("Count", fontsize=AXIS_LABEL_SIZE)
    ax.set_xlim(-5, 5)
    ax.tick_params(axis="both", labelsize=TICK_SIZE, direction="out", length=3, width=0.9)

    for spine in ax.spines.values():
        spine.set_linewidth(1.0)

    ax.legend(frameon=False, fontsize=LEGEND_SIZE, ncol=2)

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURE_DIR / "residual_geo6_3x2_final_bigfont.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "residual_geo6_3x2_final_bigfont.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "residual_geo6_3x2_final_bigfont.svg", bbox_inches="tight")

    plt.close(fig)
    print(f"已保存残差图：{FIGURE_DIR / 'residual_geo6_3x2_final_bigfont.png'}")


# =========================
# 8. 水深图
# =========================

def plot_bathymetry() -> None:
    """绘制 6 模型 3×2 水深图。"""
    pred_file = OUTPUT_DIR / "image_predictions_final.csv"
    if not pred_file.exists():
        raise FileNotFoundError(f"找不到：{pred_file}")

    pred = pd.read_csv(pred_file)
    image_raw = pd.read_csv(IMAGE_FILE)

    check_columns(pred, ["Longitude", "Latitude"] + MODEL_ORDER, "image_predictions_final.csv")
    pred = to_numeric_df(pred, ["Longitude", "Latitude"] + MODEL_ORDER)

    mask = water_mask(image_raw)
    if len(mask) != len(pred):
        raise ValueError("水体掩膜长度与 image_predictions_final.csv 行数不一致，请确认 data/out_image31.csv 没被替换。")

    for model_key in MODEL_ORDER:
        pred.loc[~mask, model_key] = np.nan

    fig, axes = plt.subplots(3, 2, figsize=(13.2, 16.0))
    axes = axes.ravel()

    cmap = "jet_r"
    norm = mcolors.Normalize(vmin=DEPTH_MIN, vmax=DEPTH_MAX)
    ticks = np.arange(DEPTH_MIN, DEPTH_MAX + 0.001, 1)

    for i, model_key in enumerate(MODEL_ORDER):
        ax = axes[i]

        lon, lat, z = pivot_map(pred, model_key)
        xx, yy = np.meshgrid(lon, lat)

        ax.pcolormesh(
            xx,
            yy,
            z,
            cmap=cmap,
            norm=norm,
            shading="auto",
            rasterized=False
        )

        ax.text(
            0.03, 0.97,
            f"({chr(97+i)})",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=PANEL_LABEL_SIZE,
            fontweight="bold"
        )

        ax.set_title(MODEL_TITLES[model_key], fontsize=TITLE_SIZE, fontweight="bold", pad=8)
        ax.grid(True, linestyle=":", linewidth=0.6, color="gray", alpha=0.75)

        xticks = interior_ticks(float(lon.min()), float(lon.max()), 3)
        yticks = interior_ticks(float(lat.min()), float(lat.max()), 3)

        ax.set_xticks(xticks)
        ax.set_yticks(yticks)

        ax.set_xticklabels(
            [format_lon(x) for x in xticks],
            fontsize=TICK_SIZE,
            rotation=0,
            ha="center"
        )
        ax.set_yticklabels(
            [format_lat(y) for y in yticks],
            fontsize=TICK_SIZE,
            rotation=90,
            va="center"
        )

        ax.set_xlabel("Longitude", fontsize=AXIS_LABEL_SIZE, labelpad=3)
        ax.set_ylabel("Latitude", fontsize=AXIS_LABEL_SIZE, labelpad=4)

        ax.tick_params(axis="x", direction="out", length=2.8, width=0.8, pad=2)
        ax.tick_params(axis="y", direction="out", length=2.8, width=0.8, pad=2)

        for spine in ax.spines.values():
            spine.set_linewidth(1.0)

        add_colorbar(fig, ax, cmap, norm, "Depth (m)", ticks)

    fig.subplots_adjust(
        left=0.08,
        right=0.965,
        bottom=0.055,
        top=0.965,
        wspace=0.44,
        hspace=0.46
    )

    FIGURE_DIR.mkdir(parents=True, exist_ok=True)

    fig.savefig(FIGURE_DIR / "bathymetry_geo6_3x2_final_bigfont.png", dpi=DPI, bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "bathymetry_geo6_3x2_final_bigfont.pdf", bbox_inches="tight")
    fig.savefig(FIGURE_DIR / "bathymetry_geo6_3x2_final_bigfont.svg", bbox_inches="tight")

    plt.close(fig)
    print(f"已保存水深图：{FIGURE_DIR / 'bathymetry_geo6_3x2_final_bigfont.png'}")


# =========================
# 9. 主函数
# =========================

def main() -> None:
    print("========== 开始绘图 ==========")
    print("模型：Stumpf / Log-linear / Random Forest / LightGBM / Geo_RF / Geo_LightGBM")
    print("输出：bigfont 版本 PNG / PDF / SVG")

    plot_scatter()
    plot_residual()
    plot_bathymetry()

    print("========== 全部完成 ==========")
    print(f"输出目录：{FIGURE_DIR}")


if __name__ == "__main__":
    main()
