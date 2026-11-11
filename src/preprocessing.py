import json
import re
from datetime import datetime
from io import StringIO
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
OUTPUTS_DIR = ROOT / "outputs"

RAW_PATH = DATA_DIR / "ispu_raw.xls"
PROCESSED_PATH = DATA_DIR / "ispu_processed.csv"
LOG_PATH = OUTPUTS_DIR / "preprocessing_log.json"

TANGGAL = "tanggal"
STASIUN = "stasiun"
TARGET = "pm25_target"
SPLIT = "split"

HISTORY_FEATURES = ["lag_1", "lag_3", "lag_7", "rolling_mean_7"]
NUMERIC_FEATURES = HISTORY_FEATURES + ["bulan"]
FEATURES = [STASIUN] + NUMERIC_FEATURES

SPLIT_LABELS = ("train", "validation", "test")
SPLIT_RATIOS = {"train": 0.70, "validation": 0.15, "test": 0.15}

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


def json_default(value):
    return value.item() if hasattr(value, "item") else str(value)


def load_raw_data(path=RAW_PATH):
    if not path.exists():
        raise FileNotFoundError(f"File tidak ditemukan: {path}")

    # File .xls ini isinya tabel HTML, bukan workbook Excel.
    tables = pd.read_html(StringIO(path.read_text(encoding="utf-8")))
    df = tables[0].copy()
    df.columns = [str(col).strip().lower() for col in df.columns]

    missing = REQUIRED_COLUMNS.difference(df.columns)
    if missing:
        raise ValueError("Kolom tidak ditemukan: " + ", ".join(sorted(missing)))

    return df


def standardize_station(value):
    text = " ".join(str(value).upper().split())
    match = re.match(r"^DKI\s*([1-5])\b", text)
    if not match:
        return None
    return STATION_NAMES[f"DKI{match.group(1)}"]


def clean_records(df_raw, log):
    df = df_raw.copy()
    df["stasiun_raw"] = df["stasiun"].astype(str).str.strip()
    df["stasiun"] = df["stasiun_raw"].map(standardize_station)

    if df["stasiun"].isna().any():
        unknown = sorted(df.loc[df["stasiun"].isna(), "stasiun_raw"].unique())
        raise ValueError("Stasiun tidak dikenal: " + ", ".join(unknown))

    for col in NUMERIC_SOURCE_COLUMNS:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    periode = pd.to_numeric(df["periode_data"], errors="coerce").astype("Int64")
    hari = pd.to_numeric(df["tanggal"], errors="coerce").astype("Int64")
    df["tanggal_dt"] = pd.to_datetime(
        periode.astype(str) + hari.astype(str).str.zfill(2),
        format="%Y%m%d",
        errors="coerce",
    )

    invalid_rows = df[df["tanggal_dt"].isna()]
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

    df = df[df["tanggal_dt"].notna()].copy()
    df[TANGGAL] = df["tanggal_dt"]
    df = df.drop(columns=["tanggal_dt"])

    duplicate_mask = df.duplicated(subset=[STASIUN, TANGGAL], keep="first")
    duplicate_rows = df[duplicate_mask]
    df = df[~duplicate_mask].copy()

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
            col: int(df[col].isna().sum()) for col in NUMERIC_SOURCE_COLUMNS
        },
        "rentang_tanggal_setelah_cleaning": {
            "mulai": str(df[TANGGAL].min().date()),
            "selesai": str(df[TANGGAL].max().date()),
        },
        "jumlah_baris_setelah_cleaning": int(len(df)),
    }

    if len(duplicate_rows):
        print(f"{len(duplicate_rows)} duplikasi stasiun-tanggal dibuang")

    return df, log


def build_daily_panel(df_clean, log):
    start = df_clean[TANGGAL].min()
    end = df_clean[TANGGAL].max()
    dates = pd.date_range(start, end, freq="D")
    stations = sorted(df_clean[STASIUN].unique())

    grid = pd.MultiIndex.from_product(
        [stations, dates], names=[STASIUN, TANGGAL]
    ).to_frame(index=False)

    panel = grid.merge(
        df_clean.drop(columns=["stasiun_raw"]),
        on=[STASIUN, TANGGAL],
        how="left",
        validate="one_to_one",
    ).sort_values([STASIUN, TANGGAL], ignore_index=True)

    missing = grid.merge(
        df_clean[[STASIUN, TANGGAL]],
        on=[STASIUN, TANGGAL],
        how="left",
        indicator=True,
    )
    missing = missing[missing["_merge"] == "left_only"]

    log["kalender"] = {
        "jumlah_tanggal_unik": int(len(dates)),
        "tanggal_mulai": str(start.date()),
        "tanggal_selesai": str(end.date()),
        "jumlah_stasiun": int(len(stations)),
        "daftar_stasiun": stations,
        "jumlah_hari_stasiun_diharapkan": int(len(grid)),
        "jumlah_hari_stasiun_teramati": int(len(df_clean)),
        "jumlah_pm25_valid": int(panel["pm_duakomalima"].notna().sum()),
        "jumlah_hari_stasiun_hilang": int(len(grid) - len(df_clean)),
        "contoh_hari_stasiun_hilang": [
            {"stasiun": row.stasiun, "tanggal": str(row.tanggal.date())}
            for row in missing[[STASIUN, TANGGAL]].head(20).itertuples(index=False)
        ],
    }

    return panel, log


def add_split(panel, log):
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

    split_by_date = {
        **{date: "train" for date in train_dates},
        **{date: "validation" for date in validation_dates},
        **{date: "test" for date in test_dates},
    }

    panel = panel.copy()
    panel[SPLIT] = panel[TANGGAL].map(split_by_date)
    if panel[SPLIT].isna().any():
        raise ValueError("Ada tanggal yang tidak mendapat label split.")

    log["split"] = {
        "rasio": SPLIT_RATIOS,
        "jumlah_tanggal_unik": n_dates,
        "train": {
            "jumlah_tanggal": n_train,
            "mulai": str(min(train_dates).date()),
            "selesai": str(max(train_dates).date()),
        },
        "validation": {
            "jumlah_tanggal": n_validation,
            "mulai": str(min(validation_dates).date()),
            "selesai": str(max(validation_dates).date()),
        },
        "test": {
            "jumlah_tanggal": n_test,
            "mulai": str(min(test_dates).date()),
            "selesai": str(max(test_dates).date()),
        },
    }

    return panel, log


def add_features(panel, log):
    df = panel.sort_values([STASIUN, TANGGAL]).copy()
    grouped = df.groupby(STASIUN, sort=False)["pm_duakomalima"]

    df["lag_1"] = grouped.shift(1)
    df["lag_3"] = grouped.shift(3)
    df["lag_7"] = grouped.shift(7)
    df["rolling_mean_7"] = grouped.transform(
        lambda series: series.shift(1).rolling(7, min_periods=7).mean()
    )
    df["bulan"] = df[TANGGAL].dt.month
    df[TARGET] = df["pm_duakomalima"]

    target_missing = df[TARGET].isna()
    history_missing = (~target_missing) & df[HISTORY_FEATURES].isna().any(axis=1)
    eligible = ~target_missing & ~history_missing

    dropped = df[~eligible].copy()
    dropped["alasan"] = np.select(
        [target_missing[~eligible], history_missing[~eligible]],
        ["Target PM2.5 kosong", "Riwayat historis tidak lengkap"],
        default="Lainnya",
    )
    dropped["kolom_riwayat_kosong"] = dropped[HISTORY_FEATURES].apply(
        lambda row: ", ".join(row.index[row.isna()]), axis=1
    )

    model_df = df.loc[
        eligible, [TANGGAL, STASIUN, *NUMERIC_FEATURES, TARGET, SPLIT]
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
        "contoh_dibuang": [
            {
                "tanggal": str(row.tanggal.date()),
                "stasiun": row.stasiun,
                "alasan": row.alasan,
                "kolom_riwayat_kosong": row.kolom_riwayat_kosong,
            }
            for row in dropped.head(12).itertuples(index=False)
        ],
        "jumlah_sampel_siap_model": int(len(model_df)),
        "rentang_sampel": {
            "mulai": str(model_df[TANGGAL].min().date()),
            "selesai": str(model_df[TANGGAL].max().date()),
        },
    }

    for label in SPLIT_LABELS:
        subset = model_df[model_df[SPLIT] == label]
        log["split"][label]["jumlah_baris_setelah_fitur"] = int(len(subset))
        log["split"][label]["rentang_baris_setelah_fitur"] = {
            "mulai": str(subset[TANGGAL].min().date()),
            "selesai": str(subset[TANGGAL].max().date()),
        }

    return model_df, log


def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    log = {
        "dibuat_pada": datetime.now().astimezone().isoformat(timespec="seconds"),
        "file_sumber": str(RAW_PATH.relative_to(ROOT)),
    }

    print(f"Membaca data mentah: {RAW_PATH}")
    df_raw = load_raw_data()
    print(f"{len(df_raw)} baris x {df_raw.shape[1]} kolom")

    print("Membersihkan data")
    df_clean, log = clean_records(df_raw, log)
    cleaning = log["cleaning"]
    print(
        f"Tanggal tidak valid: {len(cleaning['tanggal_tidak_valid'])} | "
        f"Duplikasi: {cleaning['duplikasi_stasiun_tanggal_dibuang']} | "
        f"PM2.5 kosong: {cleaning['pm25_kosong_mentah']}"
    )

    print("Membuat kalender dan fitur")
    panel, log = build_daily_panel(df_clean, log)
    panel, log = add_split(panel, log)
    model_df, log = add_features(panel, log)

    print(f"Hari stasiun hilang: {log['kalender']['jumlah_hari_stasiun_hilang']}")
    print(f"Sampel siap model: {log['fitur']['jumlah_sampel_siap_model']}")
    print(
        f"Dibuang: target kosong={log['fitur']['target_kosong']}, "
        f"riwayat tidak lengkap={log['fitur']['riwayat_tidak_lengkap']}"
    )

    for label in SPLIT_LABELS:
        info = log["split"][label]
        print(
            f"{label}: {info['jumlah_tanggal']} tanggal "
            f"({info['mulai']} s.d. {info['selesai']}), "
            f"{info['jumlah_baris_setelah_fitur']} baris"
        )

    model_df.to_csv(
        PROCESSED_PATH,
        index=False,
        date_format="%Y-%m-%d",
        float_format="%.6g",
    )
    with LOG_PATH.open("w", encoding="utf-8") as file:
        json.dump(log, file, ensure_ascii=False, indent=2, default=json_default)

    print(f"Data siap model: {PROCESSED_PATH}")
    print(f"Log: {LOG_PATH}")


if __name__ == "__main__":
    main()
