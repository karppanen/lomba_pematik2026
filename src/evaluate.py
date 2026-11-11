import json
from datetime import datetime
from pathlib import Path

import joblib
import matplotlib

matplotlib.use("Agg")

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"

PROCESSED_PATH = DATA_DIR / "ispu_processed.csv"
MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.csv"
STATION_METRICS_PATH = OUTPUTS_DIR / "metrics_per_station.csv"
SUMMARY_PATH = OUTPUTS_DIR / "run_summary.json"

ACTUAL_VS_PREDICTED_PATH = FIGURES_DIR / "actual_vs_predicted.png"
COMPARISON_PATH = FIGURES_DIR / "model_comparison.png"
IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"

TANGGAL = "tanggal"
STASIUN = "stasiun"
TARGET = "pm25_target"
SPLIT = "split"
FEATURES = ["stasiun", "lag_1", "lag_3", "lag_7", "rolling_mean_7", "bulan"]

REPRESENTATIVE_STATION = "DKI1 Bundaran HI"
FIGURE_DPI = 200

COLORS = {
    "actual": "#111827",
    "naive": "#d97706",
    "xgboost": "#2563eb",
}

BULAN_ID = {
    1: "Jan",
    2: "Feb",
    3: "Mar",
    4: "Apr",
    5: "Mei",
    6: "Jun",
    7: "Jul",
    8: "Agu",
    9: "Sep",
    10: "Okt",
    11: "Nov",
    12: "Des",
}


def format_tanggal(value):
    return f"{value.day} {BULAN_ID[value.month]} {value.year}"


def metrics(actual, predicted):
    return {
        "MAE": float(mean_absolute_error(actual, predicted)),
        "RMSE": float(np.sqrt(mean_squared_error(actual, predicted))),
    }


def load_data():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Data belum ada: {PROCESSED_PATH}. Jalankan src/preprocessing.py dulu."
        )
    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model belum ada: {MODEL_PATH}. Jalankan src/train_model.py dulu."
        )

    df = pd.read_csv(PROCESSED_PATH, parse_dates=[TANGGAL])
    bundle = joblib.load(MODEL_PATH)

    if bundle["features"] != FEATURES:
        raise ValueError(
            "Fitur model berbeda dengan script ini. Latih ulang model."
        )

    return df, bundle


def make_predictions(df, bundle):
    test = df[df[SPLIT] == "test"].copy().sort_values([TANGGAL, STASIUN])
    if test.empty:
        raise ValueError("Test set kosong.")

    if test[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Test set masih ada nilai kosong.")

    pred = bundle["pipeline"].predict(test[FEATURES])
    # Indeks ISPU berada pada rentang 0-500.
    pred = np.clip(pred, 0, 500)

    predictions = pd.DataFrame(
        {
            TANGGAL: test[TANGGAL].to_numpy(),
            STASIUN: test[STASIUN].to_numpy(),
            "actual": test[TARGET].to_numpy(),
            "naive_pred": test["lag_1"].to_numpy(),
            "xgboost_pred": pred,
        }
    ).sort_values([TANGGAL, STASIUN], ignore_index=True)

    if predictions[["actual", "naive_pred", "xgboost_pred"]].isna().any().any():
        raise ValueError("Prediksi masih ada nilai kosong.")

    return predictions


def calc_metrics(predictions):
    naive = metrics(predictions["actual"], predictions["naive_pred"])
    xgboost = metrics(predictions["actual"], predictions["xgboost_pred"])

    table = pd.DataFrame(
        [
            {"model": "Naive Forecast", **naive},
            {"model": "XGBoost", **xgboost},
        ]
    )
    return table, naive, xgboost


def station_metrics(predictions):
    rows = []
    for station, group in predictions.groupby(STASIUN, sort=True):
        naive = metrics(group["actual"], group["naive_pred"])
        xgboost = metrics(group["actual"], group["xgboost_pred"])
        reduction = (
            (naive["MAE"] - xgboost["MAE"]) / naive["MAE"] * 100
            if naive["MAE"] > 0
            else np.nan
        )
        rows.append(
            {
                "stasiun": station,
                "jumlah_observasi": int(len(group)),
                "naive_MAE": naive["MAE"],
                "naive_RMSE": naive["RMSE"],
                "xgboost_MAE": xgboost["MAE"],
                "xgboost_RMSE": xgboost["RMSE"],
                "penurunan_MAE_persen": float(reduction),
            }
        )
    return pd.DataFrame(rows)


def style_axis(axis):
    axis.spines["top"].set_visible(False)
    axis.spines["right"].set_visible(False)
    axis.grid(axis="both", color="#e5e7eb", linewidth=0.7)
    axis.set_axisbelow(True)


def plot_predictions(predictions, naive, xgboost):
    dates = predictions[TANGGAL]
    start, end = dates.min(), dates.max()
    n_observations = len(predictions)
    n_stations = predictions[STASIUN].nunique()

    fig, (ax_top, ax_bottom) = plt.subplots(
        2,
        1,
        figsize=(12.5, 8.2),
        sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.05]},
    )

    daily = (
        predictions.groupby(TANGGAL)[["actual", "naive_pred", "xgboost_pred"]]
        .mean()
        .sort_index()
    )
    ax_top.plot(
        daily.index,
        daily["actual"],
        color=COLORS["actual"],
        linewidth=2.2,
        label="Aktual",
    )
    ax_top.plot(
        daily.index,
        daily["naive_pred"],
        color=COLORS["naive"],
        linewidth=1.4,
        alpha=0.9,
        label="Naive Forecast",
    )
    ax_top.plot(
        daily.index,
        daily["xgboost_pred"],
        color=COLORS["xgboost"],
        linewidth=1.7,
        label="XGBoost",
    )
    ax_top.set_title("A. Rata-rata harian seluruh stasiun", loc="left")
    ax_top.set_ylabel("Indeks PM2.5")
    ax_top.legend(loc="upper left", ncol=3)
    ax_top.text(
        0.995,
        0.96,
        f"Test: {n_observations:,} observasi | {n_stations} stasiun\n"
        f"MAE Naive {naive['MAE']:.2f} | MAE XGBoost {xgboost['MAE']:.2f}",
        transform=ax_top.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f9fafb",
            "edgecolor": "#e5e7eb",
            "alpha": 0.95,
        },
    )

    station = (
        REPRESENTATIVE_STATION
        if REPRESENTATIVE_STATION in set(predictions[STASIUN])
        else sorted(predictions[STASIUN].unique())[0]
    )
    subset = predictions[predictions[STASIUN] == station].sort_values(TANGGAL)
    station_naive = metrics(subset["actual"], subset["naive_pred"])
    station_xgboost = metrics(subset["actual"], subset["xgboost_pred"])

    ax_bottom.plot(
        subset[TANGGAL],
        subset["actual"],
        color=COLORS["actual"],
        linewidth=2.0,
        label="Aktual",
    )
    ax_bottom.plot(
        subset[TANGGAL],
        subset["naive_pred"],
        color=COLORS["naive"],
        linewidth=1.3,
        alpha=0.9,
        label="Naive Forecast",
    )
    ax_bottom.plot(
        subset[TANGGAL],
        subset["xgboost_pred"],
        color=COLORS["xgboost"],
        linewidth=1.6,
        label="XGBoost",
    )
    ax_bottom.set_title(f"B. Stasiun representatif: {station}", loc="left")
    ax_bottom.set_ylabel("Indeks PM2.5")
    ax_bottom.set_xlabel("Tanggal (test set)")
    ax_bottom.text(
        0.995,
        0.96,
        f"MAE Naive {station_naive['MAE']:.2f} | "
        f"MAE XGBoost {station_xgboost['MAE']:.2f}",
        transform=ax_bottom.transAxes,
        ha="right",
        va="top",
        fontsize=9,
        color="#374151",
        bbox={
            "boxstyle": "round,pad=0.35",
            "facecolor": "#f9fafb",
            "edgecolor": "#e5e7eb",
            "alpha": 0.95,
        },
    )

    for axis in (ax_top, ax_bottom):
        style_axis(axis)
        low, high = axis.get_ylim()
        axis.set_ylim(low, high + (high - low) * 0.10)

    ax_bottom.xaxis.set_major_locator(mdates.DayLocator(interval=14))
    ax_bottom.xaxis.set_major_formatter(mdates.DateFormatter("%d/%m"))
    ax_bottom.set_xlim(start - pd.Timedelta(days=1), end + pd.Timedelta(days=1))

    fig.suptitle(
        "Aktual vs Prediksi Indeks PM2.5 pada Test Set "
        f"({format_tanggal(start)} s.d. {format_tanggal(end)})",
        fontsize=14,
        fontweight="bold",
        x=0.06,
        ha="left",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.95))
    fig.savefig(
        ACTUAL_VS_PREDICTED_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Grafik: {ACTUAL_VS_PREDICTED_PATH}")


def plot_comparison(naive, xgboost):
    names = ["MAE", "RMSE"]
    naive_values = [naive[name] for name in names]
    xgboost_values = [xgboost[name] for name in names]

    fig, axis = plt.subplots(figsize=(8.8, 5.6))
    positions = np.arange(len(names))
    width = 0.34

    bars_naive = axis.bar(
        positions - width / 2,
        naive_values,
        width,
        label="Naive Forecast",
        color=COLORS["naive"],
        edgecolor="white",
    )
    bars_xgboost = axis.bar(
        positions + width / 2,
        xgboost_values,
        width,
        label="XGBoost",
        color=COLORS["xgboost"],
        edgecolor="white",
    )

    for bars in (bars_naive, bars_xgboost):
        axis.bar_label(
            bars,
            fmt="%.2f",
            padding=4,
            fontsize=10,
            fontweight="bold",
            color="#111827",
        )

    axis.set_xticks(positions, names)
    axis.set_ylabel("Error (poin indeks PM2.5)")
    axis.set_title("Perbandingan MAE dan RMSE pada Test Set", loc="left")
    axis.legend(loc="upper right")
    axis.set_ylim(0, max(naive_values + xgboost_values) * 1.22)

    reduction_mae = (naive["MAE"] - xgboost["MAE"]) / naive["MAE"] * 100
    reduction_rmse = (naive["RMSE"] - xgboost["RMSE"]) / naive["RMSE"] * 100
    axis.text(
        0.5,
        0.94,
        f"XGBoost menurunkan MAE {reduction_mae:.1f}% "
        f"dan RMSE {reduction_rmse:.1f}%",
        transform=axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color="#1f2937",
        bbox={
            "boxstyle": "round,pad=0.4",
            "facecolor": "#eff6ff",
            "edgecolor": "#bfdbfe",
        },
    )
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(
        COMPARISON_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Grafik: {COMPARISON_PATH}")


def plot_importance(bundle):
    pipeline = bundle["pipeline"]
    preprocessor = pipeline.named_steps["preprocessor"]
    model = pipeline.named_steps["regressor"]

    names = np.asarray(preprocessor.get_feature_names_out())
    importances = np.asarray(model.feature_importances_)
    if len(names) != len(importances):
        raise ValueError("Jumlah nama fitur dan importance tidak sama.")

    order = np.argsort(importances)
    top_n = min(15, len(order))
    selected = order[-top_n:]
    labels = [name.replace("stasiun_", "") for name in names[selected]]
    values = importances[selected]

    fig, axis = plt.subplots(figsize=(9.2, 6.0))
    bars = axis.barh(labels, values, color="#2563eb", edgecolor="white")
    axis.bar_label(
        bars,
        fmt="%.3f",
        padding=4,
        fontsize=9,
        color="#111827",
    )
    axis.set_xlabel("Gain (kontribusi relatif)")
    axis.set_title("Kontribusi Fitur Model XGBoost", loc="left")
    axis.set_xlim(0, max(values) * 1.18)
    style_axis(axis)
    fig.tight_layout()
    fig.savefig(
        IMPORTANCE_PATH,
        dpi=FIGURE_DPI,
        bbox_inches="tight",
        facecolor="white",
    )
    plt.close(fig)
    print(f"Grafik: {IMPORTANCE_PATH}")


def main():
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.titleweight": "bold",
            "axes.labelsize": 10,
            "axes.edgecolor": "#9ca3af",
            "axes.linewidth": 0.8,
            "legend.frameon": False,
        }
    )

    df, bundle = load_data()
    predictions = make_predictions(df, bundle)
    table, naive, xgboost = calc_metrics(predictions)
    per_station = station_metrics(predictions)

    predictions.to_csv(
        PREDICTIONS_PATH,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.4f",
    )
    table.to_csv(METRICS_PATH, index=False, float_format="%.4f")
    per_station.to_csv(STATION_METRICS_PATH, index=False, float_format="%.4f")

    improvement = {
        "MAE_persen": float((naive["MAE"] - xgboost["MAE"]) / naive["MAE"] * 100),
        "RMSE_persen": float(
            (naive["RMSE"] - xgboost["RMSE"]) / naive["RMSE"] * 100
        ),
    }
    summary = {
        "dibuat_pada": datetime.now().astimezone().isoformat(timespec="seconds"),
        "test": {
            "jumlah_baris": int(len(predictions)),
            "jumlah_tanggal": int(predictions[TANGGAL].nunique()),
            "jumlah_stasiun": int(predictions[STASIUN].nunique()),
            "mulai": str(predictions[TANGGAL].min().date()),
            "selesai": str(predictions[TANGGAL].max().date()),
            "rata_rata_aktual": float(predictions["actual"].mean()),
            "standar_deviasi_aktual": float(predictions["actual"].std(ddof=1)),
        },
        "metrics": {
            "Naive Forecast": naive,
            "XGBoost": xgboost,
        },
        "penurunan_error_xgboost_persen": improvement,
        "model": {
            "best_params": bundle["best_params"],
            "best_iteration": bundle["best_iteration"],
            "rounds_refit": bundle["rounds_refit"],
            "refit_on_train_val": bundle["refit_on_train_val"],
            "trained_on": bundle["trained_on"],
            "validation_metrics": bundle["validation_metrics"],
            "trained_at": bundle["trained_at"],
            "versions": bundle["versions"],
        },
    }
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(table.to_string(index=False))
    print(
        f"Penurunan error: MAE {improvement['MAE_persen']:.1f}% | "
        f"RMSE {improvement['RMSE_persen']:.1f}%"
    )

    plot_predictions(predictions, naive, xgboost)
    plot_comparison(naive, xgboost)
    plot_importance(bundle)

    print(f"Prediksi: {PREDICTIONS_PATH}")
    print(f"Metrik: {METRICS_PATH}")
    print(f"Metrik per stasiun: {STATION_METRICS_PATH}")
    print(f"Ringkasan: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
