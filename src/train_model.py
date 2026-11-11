import json
import time
from datetime import datetime
from itertools import product
from pathlib import Path

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

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS_DIR = ROOT / "models"
OUTPUTS_DIR = ROOT / "outputs"

PROCESSED_PATH = DATA_DIR / "ispu_processed.csv"
MODEL_PATH = MODELS_DIR / "xgboost_model.joblib"
TUNING_PATH = OUTPUTS_DIR / "tuning_results.csv"
SUMMARY_PATH = OUTPUTS_DIR / "training_summary.json"

TANGGAL = "tanggal"
STASIUN = "stasiun"
TARGET = "pm25_target"
SPLIT = "split"

CATEGORICAL_FEATURES = [STASIUN]
NUMERIC_FEATURES = ["lag_1", "lag_3", "lag_7", "rolling_mean_7", "bulan"]
FEATURES = CATEGORICAL_FEATURES + NUMERIC_FEATURES

SPLIT_LABELS = ("train", "validation", "test")
RANDOM_STATE = 42

# Model final dilatih ulang pada train+validation.
REFIT_ON_TRAIN_VAL = True

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


def json_default(value):
    return value.item() if hasattr(value, "item") else str(value)


def metrics(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    return {"MAE": float(mae), "RMSE": float(rmse)}


def load_data():
    if not PROCESSED_PATH.exists():
        raise FileNotFoundError(
            f"Data belum ada: {PROCESSED_PATH}. Jalankan src/preprocessing.py dulu."
        )

    df = pd.read_csv(PROCESSED_PATH, parse_dates=[TANGGAL])
    required = {TANGGAL, STASIUN, TARGET, SPLIT, *FEATURES}
    missing = required.difference(df.columns)
    if missing:
        raise ValueError("Kolom tidak ditemukan: " + ", ".join(sorted(missing)))

    unknown = set(df[SPLIT].unique()).difference(SPLIT_LABELS)
    if unknown:
        raise ValueError(f"Label split tidak dikenal: {sorted(unknown)}")

    if df[FEATURES + [TARGET]].isna().any().any():
        raise ValueError("Data processed masih ada nilai kosong.")

    return df


def split_data(df):
    train = df[df[SPLIT] == "train"].copy()
    validation = df[df[SPLIT] == "validation"].copy()
    test = df[df[SPLIT] == "test"].copy()

    frames = {"train": train, "validation": validation, "test": test}
    for first, second in [
        ("train", "validation"),
        ("train", "test"),
        ("validation", "test"),
    ]:
        overlap = set(frames[first][TANGGAL]) & set(frames[second][TANGGAL])
        if overlap:
            raise ValueError(f"Tanggal {first} dan {second} tumpang tindih.")

    return train, validation, test


def make_preprocessor():
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


def make_pipeline(model):
    return Pipeline(
        [
            ("preprocessor", make_preprocessor()),
            ("regressor", model),
        ]
    )


def tune(train, validation):
    preprocessor = make_preprocessor()
    x_train = preprocessor.fit_transform(train[FEATURES])
    x_validation = preprocessor.transform(validation[FEATURES])
    y_train = train[TARGET].to_numpy()
    y_validation = validation[TARGET].to_numpy()

    keys = list(TUNING_GRID)
    candidates = [
        {**dict(zip(keys, values)), **FIXED_XGB_PARAMS}
        for values in product(*(TUNING_GRID[key] for key in keys))
    ]

    rows = []
    best = None

    print(f"Menguji {len(candidates)} kandidat hyperparameter")
    for number, params in enumerate(candidates, start=1):
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

        pred = model.predict(x_validation)
        score = metrics(validation[TARGET], pred)
        row = {
            **params,
            "n_estimators_maks": MAX_BOOSTING_ROUNDS,
            "best_iteration": int(model.best_iteration),
            "val_MAE": score["MAE"],
            "val_RMSE": score["RMSE"],
        }
        rows.append(row)

        print(
            f"[{number:02d}/{len(candidates)}] depth={params['max_depth']} "
            f"lr={params['learning_rate']:.2f} "
            f"mcw={params['min_child_weight']} -> "
            f"MAE={score['MAE']:.3f} RMSE={score['RMSE']:.3f} "
            f"iter={model.best_iteration} ({time.perf_counter() - started:.1f}s)"
        )

        if best is None or (score["MAE"], score["RMSE"]) < (
            best["val_MAE"],
            best["val_RMSE"],
        ):
            best = row

    results = pd.DataFrame(rows).sort_values(["val_MAE", "val_RMSE"])
    results = results.reset_index(drop=True)
    results.insert(0, "peringkat", np.arange(1, len(results) + 1))

    best_metrics = {"MAE": float(best["val_MAE"]), "RMSE": float(best["val_RMSE"])}
    return results, best, best_metrics


def main():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_data()
    train, validation, test = split_data(df)
    print(f"Baris: train={len(train)}, validation={len(validation)}, test={len(test)}")

    tuning_results, best, best_metrics = tune(train, validation)

    best_params = {
        key: best[key]
        for key in [
            "max_depth",
            "learning_rate",
            "min_child_weight",
            *FIXED_XGB_PARAMS.keys(),
        ]
    }
    best_iteration = int(best["best_iteration"])
    rounds_refit = best_iteration + 1

    naive_validation = metrics(validation[TARGET], validation["lag_1"].to_numpy())

    print(
        f"Terbaik: depth={best_params['max_depth']}, "
        f"lr={best_params['learning_rate']:.2f}, "
        f"mcw={best_params['min_child_weight']}, rounds={rounds_refit}"
    )
    print(
        f"Validation MAE: XGBoost={best_metrics['MAE']:.4f}, "
        f"Naive={naive_validation['MAE']:.4f}"
    )

    if REFIT_ON_TRAIN_VAL:
        final_train = pd.concat([train, validation], ignore_index=True)
        trained_on = "train+validation"
    else:
        final_train = train.copy()
        trained_on = "train"

    print(f"Melatih model final pada {trained_on} ({len(final_train)} baris)")
    final_model = XGBRegressor(
        objective="reg:squarederror",
        n_estimators=rounds_refit,
        tree_method="hist",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        **best_params,
    )
    pipeline = make_pipeline(final_model)
    pipeline.fit(final_train[FEATURES], final_train[TARGET])

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
        "pipeline": pipeline,
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
    tuning_results.to_csv(TUNING_PATH, index=False, float_format="%.6f")

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
    with SUMMARY_PATH.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2, default=json_default)

    print(tuning_results.head(5).to_string(index=False))
    print(f"Model: {MODEL_PATH}")
    print(f"Hasil tuning: {TUNING_PATH}")
    print(f"Ringkasan: {SUMMARY_PATH}")


if __name__ == "__main__":
    main()
