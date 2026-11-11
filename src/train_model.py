"""Pelatihan dan validasi model XGBoost.

Alur:
1. Membaca data hasil preprocessing.
2. Memisahkan train/validation/test berdasarkan kolom split.
3. Mencari hyperparameter dengan validation set (test tidak disentuh).
4. Melatih ulang model final pada train+validation.
5. Menyimpan pipeline lengkap (encoder + model) ke models/xgboost_model.joblib.

Jalankan dari folder root:
    python src/train_model.py
"""

from __future__ import annotations

import json
import logging
import time
from datetime import datetime
from itertools import product
from typing import Any

import joblib
import numpy as np
import pandas as pd
import sklearn
import xgboost
from sklearn.compose import ColumnTransformer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from xgboost import XGBRegressor

from config import (
    CATEGORICAL_FEATURES,
    EARLY_STOPPING_ROUNDS,
    FEATURES,
    FIXED_XGB_PARAMS,
    MAX_BOOSTING_ROUNDS,
    MODEL_PATH,
    NUMERIC_FEATURES,
    PROCESSED_PATH,
    RANDOM_STATE,
    REFIT_ON_TRAIN_VAL,
    SPLIT,
    SPLIT_LABELS,
    STASIUN,
    TANGGAL,
    TARGET,
    TRAINING_SUMMARY_PATH,
    TUNING_GRID,
    TUNING_RESULTS_PATH,
    ensure_directories,
)

LOGGER = logging.getLogger("training")


def _json_default(value: Any) -> Any:
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def regression_metrics(actual: pd.Series, predicted: np.ndarray) -> dict[str, float]:
    """Menghitung MAE dan RMSE dalam satuan poin indeks."""
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {"MAE": float(mae), "RMSE": float(rmse)}


def load_processed_data(path=PROCESSED_PATH) -> pd.DataFrame:
    """Membaca data siap model dan memvalidasi kolom penting."""
    if not path.exists():
        raise FileNotFoundError(
            f"Data hasil preprocessing tidak ditemukan: {path}\n"
            "Jalankan `python src/preprocessing.py` terlebih dahulu."
        )

    df = pd.read_csv(path, parse_dates=[TANGGAL])
    required = {TANGGAL, STASIUN, TARGET, SPLIT, *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError(
            "Kolom wajib tidak ditemukan pada data processed: "
            + ", ".join(sorted(missing))
        )

    unknown_splits = set(df[SPLIT].unique()).difference(SPLIT_LABELS)
    if unknown_splits:
        raise ValueError(f"Label split tidak dikenal: {sorted(unknown_splits)}")

    for label in SPLIT_LABELS:
        if not (df[SPLIT] == label).any():
            raise ValueError(f"Split '{label}' kosong.")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Data processed masih mengandung nilai kosong.")

    return df


def split_frames(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Mengambil subset train/validation/test dan memastikan tidak tumpang tindih."""
    train = df.loc[df[SPLIT] == "train"].copy()
    validation = df.loc[df[SPLIT] == "validation"].copy()
    test = df.loc[df[SPLIT] == "test"].copy()

    date_sets = {
        label: set(frame[TANGGAL])
        for label, frame in (
            ("train", train),
            ("validation", validation),
            ("test", test),
        )
    }
    for first, second in (
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ):
        overlap = date_sets[first] & date_sets[second]
        if overlap:
            raise ValueError(
                f"Tanggal {first} dan {second} tumpang tindih: "
                f"{sorted(overlap)[:5]}"
            )

    return train, validation, test


def build_preprocessor() -> ColumnTransformer:
    """One-hot encoding stasiun + fitur numerik apa adanya."""
    return ColumnTransformer(
        transformers=[
            (
                "stasiun",
                OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                CATEGORICAL_FEATURES,
            ),
            ("numerik", "passthrough", NUMERIC_FEATURES),
        ],
        remainder="drop",
        verbose_feature_names_out=False,
    )


def build_pipeline(regressor: XGBRegressor) -> Pipeline:
    return Pipeline(
        steps=[
            ("preprocessor", build_preprocessor()),
            ("regressor", regressor),
        ]
    )


def iter_candidates() -> list[dict[str, Any]]:
    """Kombinasi hyperparameter yang akan diuji pada validation set."""
    keys = list(TUNING_GRID)
    return [
        {**dict(zip(keys, values)), **FIXED_XGB_PARAMS}
        for values in product(*(TUNING_GRID[key] for key in keys))
    ]


def tune_xgboost(
    train: pd.DataFrame, validation: pd.DataFrame
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, float], np.ndarray]:
    """Mencari konfigurasi terbaik berdasarkan validation MAE."""
    preprocessor = build_preprocessor()
    x_train = preprocessor.fit_transform(train[FEATURES])
    x_validation = preprocessor.transform(validation[FEATURES])
    y_train = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()

    candidates = iter_candidates()
    results: list[dict[str, Any]] = []
    best_row: dict[str, Any] | None = None

    LOGGER.info(
        "Menguji %d kandidat hyperparameter pada validation set",
        len(candidates),
    )
    for index, params in enumerate(candidates, start=1):
        started = time.perf_counter()
        model = XGBRegressor(
            objective="reg:squarederror",
            n_estimators=MAX_BOOSTING_ROUNDS,
            tree_method="hist",
            random_state=RANDOM_STATE,
            n_jobs=-1,
            early_stopping_rounds=EARLY_STOPPING_ROUNDS,
            eval_metric="mae",
            **params,
        )
        model.fit(
            x_train,
            y_train,
            eval_set=[(x_validation, y_validation)],
            verbose=False,
        )
        predictions = model.predict(x_validation)
        metrics = regression_metrics(validation[TARGET], predictions)
        row = {
            **params,
            "n_estimators_maks": MAX_BOOSTING_ROUNDS,
            "best_iteration": int(model.best_iteration),
            "val_MAE": metrics["MAE"],
            "val_RMSE": metrics["RMSE"],
        }
        results.append(row)
        LOGGER.info(
            "[%02d/%02d] depth=%d lr=%.2f mcw=%d -> MAE=%.3f | RMSE=%.3f "
            "| iter=%d (%.1fs)",
            index,
            len(candidates),
            params["max_depth"],
            params["learning_rate"],
            params["min_child_weight"],
            metrics["MAE"],
            metrics["RMSE"],
            model.best_iteration,
            time.perf_counter() - started,
        )

        if best_row is None or (metrics["MAE"], metrics["RMSE"]) < (
            best_row["val_MAE"],
            best_row["val_RMSE"],
        ):
            best_row = row

    if best_row is None:
        raise RuntimeError("Tidak ada kandidat model yang berhasil dilatih.")

    tuning_results = (
        pd.DataFrame(results)
        .sort_values(["val_MAE", "val_RMSE"])
        .reset_index(drop=True)
    )
    tuning_results.insert(0, "peringkat", np.arange(1, len(tuning_results) + 1))

    best_metrics = {
        "MAE": float(best_row["val_MAE"]),
        "RMSE": float(best_row["val_RMSE"]),
    }
    return tuning_results, best_row, best_metrics


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ensure_directories()

    LOGGER.info("Membaca data processed: %s", PROCESSED_PATH)
    df = load_processed_data()
    train, validation, test = split_frames(df)
    LOGGER.info(
        "Baris: train=%d, validation=%d, test=%d",
        len(train),
        len(validation),
        len(test),
    )

    tuning_results, best_row, best_metrics = tune_xgboost(train, validation)

    best_params = {
        key: best_row[key]
        for key in [
            "max_depth",
            "learning_rate",
            "min_child_weight",
            *FIXED_XGB_PARAMS.keys(),
        ]
    }
    best_iteration = int(best_row["best_iteration"])
    rounds_refit = best_iteration + 1

    naive_validation = regression_metrics(
        validation[TARGET], validation["lag_1"].to_numpy()
    )

    LOGGER.info(
        "Konfigurasi terbaik: depth=%d, lr=%.2f, mcw=%d, rounds=%d",
        best_params["max_depth"],
        best_params["learning_rate"],
        best_params["min_child_weight"],
        rounds_refit,
    )
    LOGGER.info(
        "Validation MAE: XGBoost=%.4f | Naive=%.4f",
        best_metrics["MAE"],
        naive_validation["MAE"],
    )

    if REFIT_ON_TRAIN_VAL:
        final_train = pd.concat([train, validation], ignore_index=True)
        trained_on = "train+validation"
    else:
        final_train = train.copy()
        trained_on = "train"

    LOGGER.info(
        "Melatih model final pada %s (%d baris, %d boosting rounds)",
        trained_on,
        len(final_train),
        rounds_refit,
    )
    final_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=rounds_refit,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    final_pipeline = build_pipeline(final_model)
    final_pipeline.fit(final_train[FEATURES], final_train[TARGET])

    split_summary = {
        label: {
            "jumlah_baris": int((df[SPLIT] == label).sum()),
            "jumlah_tanggal": int(df.loc[df[SPLIT] == label, TANGGAL].nunique()),
            "mulai": str(df.loc[df[SPLIT] == label, TANGGAL].min().date()),
            "selesai": str(df.loc[df[SPLIT] == label, TANGGAL].max().date()),
        }
        for label in SPLIT_LABELS
    }

    trained_at = datetime.now().astimezone().isoformat(timespec="seconds")
    bundle = {
        "pipeline": final_pipeline,
        "features": FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "target": TARGET,
        "station_column": STASIUN,
        "date_column": TANGGAL,
        "best_params": best_params,
        "best_iteration": best_iteration,
        "rounds_refit": rounds_refit,
        "refit_on_train_val": REFIT_ON_TRAIN_VAL,
        "trained_on": trained_on,
        "training_rows": int(len(final_train)),
        "validation_metrics": best_metrics,
        "naive_validation_metrics": naive_validation,
        "split_summary": split_summary,
        "trained_at": trained_at,
        "versions": {
            "xgboost": xgboost.__version__,
            "scikit_learn": sklearn.__version__,
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }

    joblib.dump(bundle, MODEL_PATH)
    tuning_results.to_csv(TUNING_RESULTS_PATH, index=False, float_format="%.6f")

    summary = {
        "dibuat_pada": trained_at,
        "data": split_summary,
        "tuning": {
            "metrik_pemilihan": "validation MAE",
            "jumlah_kandidat": int(len(tuning_results)),
            "grid": TUNING_GRID,
            "parameter_tetap": FIXED_XGB_PARAMS,
            "max_rounds": MAX_BOOSTING_ROUNDS,
            "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        },
        "model_terbaik": {
            "params": best_params,
            "best_iteration": best_iteration,
            "rounds_refit": rounds_refit,
            "validation_MAE": best_metrics["MAE"],
            "validation_RMSE": best_metrics["RMSE"],
        },
        "baseline_naive_validation": naive_validation,
        "model_final": {
            "dilatih_pada": trained_on,
            "jumlah_baris": int(len(final_train)),
            "rentang_tanggal": {
                "mulai": str(final_train[TANGGAL].min().date()),
                "selesai": str(final_train[TANGGAL].max().date()),
            },
        },
        "fitur": FEATURES,
        "versi": bundle["versions"],
    }
    with TRAINING_SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=_json_default)

    LOGGER.info("\n%s", tuning_results.head(5).to_string(index=False))
    LOGGER.info("Model disimpan ke: %s", MODEL_PATH)
    LOGGER.info("Hasil tuning disimpan ke: %s", TUNING_RESULTS_PATH)
    LOGGER.info("Ringkasan training disimpan ke: %s", TRAINING_SUMMARY_PATH)


if __name__ == "__main__":
    main()
