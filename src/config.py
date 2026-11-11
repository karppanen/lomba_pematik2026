"""Konfigurasi terpusat untuk seluruh pipeline.

Semua path diturunkan dari lokasi file ini sehingga script dapat dijalankan
dari folder root proyek tanpa bergantung pada current working directory.
"""

from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
SRC_DIR = ROOT / "src"
NOTEBOOK_DIR = ROOT / "notebooks"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"
FIGURES_DIR = OUTPUTS_DIR / "figures"
REPORT_DIR = ROOT / "report"
PRESENTATION_DIR = ROOT / "presentation"

RAW_PATH = DATA_DIR / "ispu_raw.xls"
PROCESSED_PATH = DATA_DIR / "ispu_processed.csv"
MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"

PREDICTIONS_PATH = OUTPUTS_DIR / "predictions.csv"
METRICS_PATH = OUTPUTS_DIR / "metrics.csv"
METRICS_PER_STATION_PATH = OUTPUTS_DIR / "metrics_per_station.csv"
TUNING_RESULTS_PATH = OUTPUTS_DIR / "tuning_results.csv"
TRAINING_SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"
PREPROCESSING_LOG_PATH = OUTPUTS_DIR / "preprocessing_log.json"
RUN_SUMMARY_PATH = OUTPUTS_DIR / "run_summary.json"

ACTUAL_VS_PREDICTED_PATH = FIGURES_DIR / "actual_vs_predicted.png"
MODEL_COMPARISON_PATH = FIGURES_DIR / "model_comparison.png"
FEATURE_IMPORTANCE_PATH = FIGURES_DIR / "feature_importance.png"

TANGGAL = "tanggal"
STASIUN = "stasiun"
TARGET = "pm25_target"
SPLIT = "split"

CATEGORICAL_FEATURES = [STASIUN]
HISTORY_FEATURES = ["lag_1", "lag_3", "lag_7", "rolling_mean_7"]
NUMERIC_FEATURES = [*HISTORY_FEATURES, "bulan"]
FEATURES = [*CATEGORICAL_FEATURES, *NUMERIC_FEATURES]

SPLIT_LABELS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

RANDOM_STATE = 42

# Stasiun pusat untuk panel detail pada grafik actual_vs_predicted.png.
REPRESENTATIVE_STATION = "DKI1 Bundaran HI"

# Keputusan ini diambil sebelum test set dievaluasi: setelah hyperparameter
# dipilih memakai validation, model final dilatih ulang pada train+validation.
REFIT_ON_TRAIN_VAL = True

# Kandidat hyperparameter XGBoost. Nilai tetap (subsample, colsample, lambda)
# dibatasi agar pencarian tetap kecil dan reprodusibel.
TUNING_GRID = {
    "max_depth": [2, 3, 4],
    "learning_rate": [0.03, 0.05, 0.10],
    "min_child_weight": [1, 5],
}
FIXED_XGB_PARAMS = {
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "reg_lambda": 1.0,
}
MAX_BOOSTING_ROUNDS = 1500
EARLY_STOPPING_ROUNDS = 50

FIGURE_DPI = 200


def ensure_directories() -> None:
    """Membuat folder output yang dibutuhkan pipeline."""
    for directory in (
        DATA_DIR,
        MODELS_DIR,
        OUTPUTS_DIR,
        FIGURES_DIR,
        REPORT_DIR,
        PRESENTATION_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)
