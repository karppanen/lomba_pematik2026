"""Preprocessing dataset ISPU DKI Jakarta.

Pipeline:
1. Membaca file mentah berformat HTML (walau berekstensi .xls).
2. Membersihkan tanggal, nama stasiun, tipe numerik, dan duplikasi.
3. Membangun kalender harian lengkap per stasiun.
4. Membuat fitur historis (lag dan rolling mean) tanpa data masa depan.
5. Membagi tanggal secara kronologis menjadi train/validation/test.
6. Menyimpan data siap model ke data/ispu_processed.csv beserta log.

Jalankan dari folder root:
    python src/preprocessing.py
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from config import (
    FEATURES,
    HISTORY_FEATURES,
    NUMERIC_FEATURES,
    PREPROCESSING_LOG_PATH,
    PROCESSED_PATH,
    RAW_PATH,
    SPLIT,
    SPLIT_LABELS,
    SPLIT_RATIOS,
    STASIUN,
    TANGGAL,
    TARGET,
    ensure_directories,
)

LOGGER = logging.getLogger("preprocessing")

NUMERIC_SOURCE_COLUMNS = [
    "pm_sepuluh",
    "pm_duakomalima",
    "sulfur_dioksida",
    "karbon_monoksida",
    "ozon",
    "nitrogen_dioksida",
    "max",
]

REQUIRED_COLUMNS = {
    "periode_data",
    "bulan",
    "tanggal",
    "stasiun",
    "pm_duakomalima",
    *NUMERIC_SOURCE_COLUMNS,
}

STATION_NAMES = {
    "DKI1": "DKI1 Bundaran HI",
    "DKI2": "DKI2 Kelapa Gading",
    "DKI3": "DKI3 Jagakarsa",
    "DKI4": "DKI4 Lubang Buaya",
    "DKI5": "DKI5 Kebon Jeruk",
}


def _json_default(value: Any) -> Any:
    """Konversi tipe numpy/pandas agar aman ditulis sebagai JSON."""
    if hasattr(value, "item"):
        return value.item()
    return str(value)


def load_raw_data(path: Path = RAW_PATH) -> pd.DataFrame:
    """Membaca tabel HTML dari file .xls mentah."""
    if not path.exists():
        raise FileNotFoundError(
            f"Dataset mentah tidak ditemukan: {path}\n"
            "Salin file ISPU asli ke data/ispu_raw.xls terlebih dahulu."
        )

    text = path.read_text(encoding="utf-8")
    tables = pd.read_html(StringIO(text))
    if not tables:
        raise ValueError(f"Tidak ada tabel yang bisa dibaca dari {path}.")

    df = tables[0].copy()
    df.columns = [str(column).strip().lower() for column in df.columns]

    missing_columns = REQUIRED_COLUMNS.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "Kolom wajib tidak ditemukan: " + ", ".join(sorted(missing_columns))
        )

    return df


def standardize_station(value: Any) -> str | None:
    """Memetakan variasi nama stasiun ke satu nama kanonis."""
    text = " ".join(str(value).upper().split())
    match = re.match(r"^DKI\s*([1-5])\b", text)
    if not match:
        return None
    return STATION_NAMES[f"DKI{match.group(1)}"]


def _parse_dates(df: pd.DataFrame) -> pd.Series:
    """Menggabungkan periode_data (YYYYMM) dan tanggal (DD)."""
    period = (
        pd.to_numeric(df["periode_data"], errors="coerce")
        .astype("Int64")
        .astype(str)
    )
    day = (
        pd.to_numeric(df["tanggal"], errors="coerce")
        .astype("Int64")
        .astype(str)
        .str.zfill(2)
    )
    return pd.to_datetime(period + day, format="%Y%m%d", errors="coerce")


def clean_records(
    df_raw: pd.DataFrame, log: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Membersihkan tanggal, stasiun, nilai numerik, dan duplikasi."""
    df = df_raw.copy()
    df["stasiun_raw"] = df["stasiun"].astype(str).str.strip()
    df["stasiun"] = df["stasiun_raw"].map(standardize_station)

    unknown_stations = sorted(df.loc[df["stasiun"].isna(), "stasiun_raw"].unique())
    if unknown_stations:
        raise ValueError(
            "Nama stasiun tidak dikenali: " + ", ".join(unknown_stations)
        )

    for column in NUMERIC_SOURCE_COLUMNS:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df["tanggal_dt"] = _parse_dates(df)

    invalid_rows = df.loc[df["tanggal_dt"].isna()].copy()
    invalid_dates = [
        {
            "periode_data": int(row.periode_data),
            "bulan": int(row.bulan),
            "tanggal": int(row.tanggal),
            "stasiun_raw": row.stasiun_raw,
            "stasiun": row.stasiun,
            "pm_duakomalima": (
                None if pd.isna(row.pm_duakomalima) else float(row.pm_duakomalima)
            ),
            "alasan": "Tanggal tidak ada dalam kalender",
        }
        for row in invalid_rows.itertuples(index=False)
    ]
    df = df.loc[df["tanggal_dt"].notna()].copy()
    df[TANGGAL] = df["tanggal_dt"]
    df = df.drop(columns=["tanggal_dt"])

    duplicate_mask = df.duplicated(subset=[STASIUN, TANGGAL], keep="first")
    duplicate_rows = df.loc[duplicate_mask].copy()
    df = df.loc[~duplicate_mask].copy()

    aliases = (
        df.groupby(["stasiun_raw", STASIUN], dropna=False)
        .size()
        .reset_index(name="jumlah_baris")
        .to_dict(orient="records")
    )

    log["cleaning"] = {
        "jumlah_baris_mentah": int(len(df_raw)),
        "jumlah_kolom_mentah": int(df_raw.shape[1]),
        "kolom_mentah": list(df_raw.columns),
        "variasi_nama_stasiun": aliases,
        "tanggal_tidak_valid": invalid_dates,
        "duplikasi_stasiun_tanggal_dibuang": int(len(duplicate_rows)),
        "pm25_kosong_mentah": int(df_raw["pm_duakomalima"].isna().sum()),
        "pm25_kosong_setelah_cleaning": int(df["pm_duakomalima"].isna().sum()),
        "nilai_numerik_kosong": {
            column: int(df[column].isna().sum()) for column in NUMERIC_SOURCE_COLUMNS
        },
        "rentang_tanggal_setelah_cleaning": {
            "mulai": str(df[TANGGAL].min().date()),
            "selesai": str(df[TANGGAL].max().date()),
        },
        "jumlah_baris_setelah_cleaning": int(len(df)),
    }

    if len(duplicate_rows):
        LOGGER.warning(
            "%d duplikasi stasiun-tanggal dibuang (baris pertama dipertahankan).",
            len(duplicate_rows),
        )

    return df, log


def build_daily_panel(
    df_clean: pd.DataFrame, log: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Membuat grid harian lengkap per stasiun tanpa imputasi nilai."""
    start = df_clean[TANGGAL].min()
    end = df_clean[TANGGAL].max()
    dates = pd.date_range(start, end, freq="D")
    stations = sorted(df_clean[STASIUN].unique())

    grid = (
        pd.MultiIndex.from_product(
            [stations, dates], names=[STASIUN, TANGGAL]
        )
        .to_frame(index=False)
    )
    panel = grid.merge(
        df_clean.drop(columns=["stasiun_raw"]),
        on=[STASIUN, TANGGAL],
        how="left",
        validate="one_to_one",
    ).sort_values([STASIUN, TANGGAL], ignore_index=True)

    observed_rows = int(len(df_clean))
    observed_pm25 = int(panel["pm_duakomalima"].notna().sum())
    missing_panel = grid.merge(
        df_clean[[STASIUN, TANGGAL]],
        on=[STASIUN, TANGGAL],
        how="left",
        indicator=True,
    )
    missing_panel = missing_panel.loc[missing_panel["_merge"] == "left_only"]
    missing_dates = [
        {
            "stasiun": row.stasiun,
            "tanggal": str(row.tanggal.date()),
        }
        for row in missing_panel[[STASIUN, TANGGAL]].itertuples(index=False)
    ]

    log["kalender"] = {
        "jumlah_tanggal_unik": int(len(dates)),
        "tanggal_mulai": str(start.date()),
        "tanggal_selesai": str(end.date()),
        "jumlah_stasiun": int(len(stations)),
        "daftar_stasiun": stations,
        "jumlah_hari_stasiun_diharapkan": int(len(grid)),
        "jumlah_hari_stasiun_teramati": observed_rows,
        "jumlah_pm25_valid": observed_pm25,
        "jumlah_hari_stasiun_hilang": int(len(grid) - observed_rows),
        "contoh_hari_stasiun_hilang": missing_dates[:20],
    }

    return panel, log


def add_temporal_split(
    panel: pd.DataFrame, log: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Membagi tanggal unik secara kronologis 70/15/15."""
    dates = sorted(pd.Timestamp(value) for value in panel[TANGGAL].unique())
    n_dates = len(dates)
    n_train = int(round(n_dates * SPLIT_RATIOS["train"]))
    n_validation = int(round(n_dates * SPLIT_RATIOS["validation"]))
    n_test = n_dates - n_train - n_validation

    if min(n_train, n_validation, n_test) <= 0:
        raise ValueError("Jumlah tanggal terlalu sedikit untuk split 70/15/15.")

    train_dates = set(dates[:n_train])
    validation_dates = set(dates[n_train : n_train + n_validation])
    test_dates = set(dates[n_train + n_validation :])

    if not (
        train_dates.isdisjoint(validation_dates)
        and train_dates.isdisjoint(test_dates)
        and validation_dates.isdisjoint(test_dates)
    ):
        raise ValueError("Split tanggal saling tumpang tindih.")

    date_to_split = {
        **{date: "train" for date in train_dates},
        **{date: "validation" for date in validation_dates},
        **{date: "test" for date in test_dates},
    }
    panel = panel.copy()
    panel[SPLIT] = panel[TANGGAL].map(date_to_split)

    if panel[SPLIT].isna().any():
        raise ValueError("Ada tanggal yang tidak mendapat label split.")

    split_info = {
        "rasio": SPLIT_RATIOS,
        "jumlah_tanggal_unik": int(n_dates),
        "train": {
            "jumlah_tanggal": int(n_train),
            "mulai": str(pd.Timestamp(min(train_dates)).date()),
            "selesai": str(pd.Timestamp(max(train_dates)).date()),
        },
        "validation": {
            "jumlah_tanggal": int(n_validation),
            "mulai": str(pd.Timestamp(min(validation_dates)).date()),
            "selesai": str(pd.Timestamp(max(validation_dates)).date()),
        },
        "test": {
            "jumlah_tanggal": int(n_test),
            "mulai": str(pd.Timestamp(min(test_dates)).date()),
            "selesai": str(pd.Timestamp(max(test_dates)).date()),
        },
    }
    log["split"] = split_info

    return panel, log


def add_features(
    panel: pd.DataFrame, log: dict[str, Any]
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Membuat lag/rolling per stasiun dan menyaring sampel yang layak."""
    df = panel.sort_values([STASIUN, TANGGAL]).copy()
    grouped = df.groupby(STASIUN, sort=False)["pm_duakomalima"]

    df["lag_1"] = grouped.shift(1)
    df["lag_3"] = grouped.shift(3)
    df["lag_7"] = grouped.shift(7)
    df["rolling_mean_7"] = df.groupby(STASIUN, sort=False)[
        "pm_duakomalima"
    ].transform(lambda series: series.shift(1).rolling(7, min_periods=7).mean())
    df["bulan"] = df[TANGGAL].dt.month
    df[TARGET] = df["pm_duakomalima"]

    target_missing = df[TARGET].isna()
    history_missing = (~target_missing) & df[HISTORY_FEATURES].isna().any(axis=1)
    eligible = ~target_missing & ~history_missing

    dropped = df.loc[~eligible].copy()
    dropped["alasan"] = np.select(
        [target_missing.loc[~eligible], history_missing.loc[~eligible]],
        ["Target PM2.5 kosong", "Riwayat historis tidak lengkap"],
        default="Lainnya",
    )
    dropped["kolom_riwayat_kosong"] = dropped[HISTORY_FEATURES].apply(
        lambda row: ", ".join(row.index[row.isna()]), axis=1
    )
    dropped_examples = [
        {
            "tanggal": str(row.tanggal.date()),
            "stasiun": row.stasiun,
            "alasan": row.alasan,
            "kolom_riwayat_kosong": row.kolom_riwayat_kosong,
        }
        for row in dropped.head(12).itertuples(index=False)
    ]

    model_df = df.loc[
        eligible,
        [TANGGAL, STASIUN, *NUMERIC_FEATURES, TARGET, SPLIT],
    ].copy()
    model_df["rolling_mean_7"] = model_df["rolling_mean_7"].round(2)

    if model_df[FEATURES].isna().any().any() or model_df[TARGET].isna().any():
        raise ValueError("Masih ada nilai kosong pada data siap model.")

    log["fitur"] = {
        "definisi": {
            "lag_1": "PM2.5 pada T-1",
            "lag_3": "PM2.5 pada T-3",
            "lag_7": "PM2.5 pada T-7",
            "rolling_mean_7": "Rata-rata PM2.5 T-7 sampai T-1 (tanpa T)",
            "bulan": "Bulan dari tanggal target T",
            "stasiun": "Identitas stasiun (one-hot saat training)",
        },
        "jumlah_baris_panel": int(len(df)),
        "target_kosong": int(target_missing.sum()),
        "riwayat_tidak_lengkap": int(history_missing.sum()),
        "total_dibuang": int(len(dropped)),
        "contoh_dibuang": dropped_examples,
        "jumlah_sampel_siap_model": int(len(model_df)),
        "rentang_sampel": {
            "mulai": str(model_df[TANGGAL].min().date()),
            "selesai": str(model_df[TANGGAL].max().date()),
        },
    }

    for label in SPLIT_LABELS:
        subset = model_df.loc[model_df[SPLIT] == label]
        log["split"][label]["jumlah_baris_setelah_fitur"] = int(len(subset))
        log["split"][label]["rentang_baris_setelah_fitur"] = {
            "mulai": str(subset[TANGGAL].min().date()),
            "selesai": str(subset[TANGGAL].max().date()),
        }

    return model_df, log


def build_processed_dataset() -> tuple[pd.DataFrame, dict[str, Any]]:
    """Menjalankan seluruh tahap preprocessing dan mengembalikan data + log."""
    log: dict[str, Any] = {
        "dibuat_pada": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_sumber": str(RAW_PATH.relative_to(RAW_PATH.parents[1])),
    }

    LOGGER.info("[1/4] Membaca data mentah: %s", RAW_PATH)
    df_raw = load_raw_data()
    LOGGER.info("      %d baris x %d kolom", *df_raw.shape)

    LOGGER.info("[2/4] Membersihkan tanggal, stasiun, dan nilai numerik")
    df_clean, log = clean_records(df_raw, log)
    cleaning = log["cleaning"]
    LOGGER.info(
        "      Tanggal tidak valid: %d | Duplikasi: %d | PM2.5 kosong: %d",
        len(cleaning["tanggal_tidak_valid"]),
        cleaning["duplikasi_stasiun_tanggal_dibuang"],
        cleaning["pm25_kosong_mentah"],
    )

    LOGGER.info("[3/4] Membangun kalender harian dan fitur historis")
    panel, log = build_daily_panel(df_clean, log)
    panel, log = add_temporal_split(panel, log)
    model_df, log = add_features(panel, log)
    fitur = log["fitur"]
    LOGGER.info(
        "      Hari stasiun hilang: %d | Sampel siap model: %d",
        log["kalender"]["jumlah_hari_stasiun_hilang"],
        fitur["jumlah_sampel_siap_model"],
    )
    LOGGER.info(
        "      Dibuang: target kosong=%d, riwayat tidak lengkap=%d",
        fitur["target_kosong"],
        fitur["riwayat_tidak_lengkap"],
    )

    LOGGER.info("[4/4] Split temporal berdasarkan tanggal unik")
    for label in SPLIT_LABELS:
        info = log["split"][label]
        LOGGER.info(
            "      %-10s : %d tanggal (%s s.d. %s), %d baris",
            label,
            info["jumlah_tanggal"],
            info["mulai"],
            info["selesai"],
            info["jumlah_baris_setelah_fitur"],
        )

    return model_df, log


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")
    ensure_directories()

    model_df, log = build_processed_dataset()

    model_df.to_csv(
        PROCESSED_PATH,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6g",
    )
    with PREPROCESSING_LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2, default=_json_default)

    LOGGER.info("Data siap model disimpan ke: %s", PROCESSED_PATH)
    LOGGER.info("Log preprocessing disimpan ke: %s", PREPROCESSING_LOG_PATH)


if __name__ == "__main__":
    main()
