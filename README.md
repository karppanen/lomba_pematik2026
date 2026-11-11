# Prediksi Nilai Indeks PM2.5 Harian Berdasarkan Data ISPU DKI Jakarta Menggunakan Naive Forecast dan XGBoost

Proyek ini membandingkan **Naive Forecast** sebagai baseline dengan **XGBoost Regressor** untuk memprediksi **nilai indeks PM2.5 pada hari berikutnya (H+1)** di lima stasiun pemantauan kualitas udara DKI Jakarta.

Skenario evaluasi adalah **one-step-ahead forecasting**: saat memprediksi tanggal T, nilai aktual sampai T−1 boleh dipakai sebagai fitur, tetapi nilai aktual T dan sesudahnya tidak boleh dipakai. Ini bukan peramalan banyak hari ke depan tanpa pembaruan observasi.

---

## 1. Ringkasan Hasil

Pengujian dilakukan pada **449 observasi test** (105 tanggal, 18 Agustus–30 November 2025) dengan baris yang sama untuk kedua metode.

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Forecast | 11,9978 | 15,7731 |
| **XGBoost** | **10,5579** | **13,5799** |
| Penurunan error | **12,0%** | **13,9%** |

MAE dan RMSE memakai satuan **poin indeks PM2.5**. Semakin kecil nilainya, semakin baik.

### Hasil validation (untuk pemilihan hyperparameter)

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Forecast | 15,0157 | 19,2693 |
| XGBoost | 12,4926 | 15,9400 |

Validation set tidak dipakai untuk mengevaluasi hasil akhir. Setelah hyperparameter dipilih, model final dilatih ulang pada **train+validation** dan diuji satu kali pada test set.

### Hasil per stasiun pada test set

| Stasiun | n | Naive MAE | XGBoost MAE | Naive RMSE | XGBoost RMSE | Penurunan MAE |
|---|---:|---:|---:|---:|---:|---:|
| DKI1 Bundaran HI | 97 | 10,1443 | 8,8767 | 13,7541 | 12,1113 | 12,5% |
| DKI2 Kelapa Gading | 97 | 12,3711 | 11,6651 | 16,1482 | 14,3979 | 5,7% |
| DKI3 Jagakarsa | 82 | 13,5244 | 11,3909 | 17,6597 | 14,6021 | 15,8% |
| DKI4 Lubang Buaya | 97 | 11,1959 | 9,9475 | 14,3275 | 12,4427 | 11,2% |
| DKI5 Kebon Jeruk | 76 | 13,2632 | 11,1710 | 17,2360 | 14,4931 | 15,8% |

XGBoost memiliki MAE dan RMSE lebih rendah daripada Naive Forecast pada seluruh stasiun. Penurunan terbesar terjadi di DKI3 dan DKI5; penurunan terkecil di DKI2.

### Visualisasi utama

![Aktual vs prediksi](outputs/figures/actual_vs_predicted.png)

![Perbandingan MAE dan RMSE](outputs/figures/model_comparison.png)

![Kontribusi fitur](outputs/figures/feature_importance.png)

`lag_1` dan `rolling_mean_7` mendominasi kontribusi model. Station dummy hanya membantu pada sebagian stasiun, terutama DKI3, DKI4, dan DKI5.

---

## 2. Dataset

| Komponen | Keterangan |
|---|---|
| Sumber | Dataset ISPU DKI Jakarta (file mentah `data/ispu_raw.xls`) |
| Format asli | Tabel HTML dengan ekstensi `.xls` |
| Ukuran | 3.501 baris × 13 kolom |
| Periode | 1 Januari 2024 – 30 November 2025 |
| Tanggal unik | 700 hari kalender |
| Stasiun | 5 stasiun |
| Target | `pm_duakomalima` (indeks PM2.5), bukan konsentrasi µg/m³ |

### Kualitas data

- **61 nilai PM2.5 kosong** pada data mentah (sekitar 1,7%). Nilai kosong tidak diimputasi.
- **Satu tanggal tidak valid**: 31 September 2025 (DKI5 Kebon Jeruk). Baris ini dibuang dan dicatat di `outputs/preprocessing_log.json`; tanggal penggantinya tidak ditebak.
- **Variasi nama stasiun** distandardisasi:

| Nama mentah | Nama kanonis |
|---|---|
| `DKI1 BUNDARAN HOTEL INDONESIA (HI)` | `DKI1 Bundaran HI` |
| `DKI1 BUNDARAN HOTEL INDONESIA HI` | `DKI1 Bundaran HI` |
| `DKI1 BUNDERAN HI` | `DKI1 Bundaran HI` |
| `DKI2 KELAPA GADING` | `DKI2 Kelapa Gading` |
| `DKI3 JAGAKARSA` | `DKI3 Jagakarsa` |
| `DKI4 LUBANG BUAYA` | `DKI4 Lubang Buaya` |
| `DKI5 KEBON JERUK` | `DKI5 Kebon Jeruk` |
| `DKI5 KEBON JERUK JAKARTA BARAT` | `DKI5 Kebon Jeruk` |

Setelah cleaning, seluruh 700 tanggal tersedia untuk setiap stasiun (3.500 hari-stasiun). Tidak ada hari stasiun yang hilang dari file; nilai PM2.5 yang kosong tetap dibiarkan kosong.

---

## 3. Metodologi

### 3.1 Preprocessing

1. Membaca file dengan `pandas.read_html()` karena isinya tabel HTML.
2. Menggabungkan `periode_data` (YYYYMM) dan `tanggal` (DD) menjadi `datetime`.
3. Menstandardisasi nama stasiun dan mengonversi kolom polutan menjadi numerik (`N/A` → kosong).
4. Membuang tanggal tidak valid dan duplikasi stasiun–tanggal.
5. Membangun kalender harian lengkap per stasiun tanpa imputasi.
6. Menghitung fitur historis per stasiun berdasarkan hari kalender.
7. Membagi tanggal unik secara kronologis: **70% train, 15% validation, 15% test**.
8. Menyimpan data siap model ke `data/ispu_processed.csv`.

### 3.2 Feature engineering

Fitur dihitung relatif terhadap tanggal target T, hanya dari nilai sebelum T:

| Fitur | Definisi |
|---|---|
| `stasiun` | Identitas stasiun, di-*one-hot encode* di dalam pipeline |
| `lag_1` | PM2.5 pada T−1 |
| `lag_3` | PM2.5 pada T−3 |
| `lag_7` | PM2.5 pada T−7 |
| `rolling_mean_7` | Rata-rata PM2.5 T−7 sampai T−1 (tanpa T) |
| `bulan` | Bulan dari tanggal target T |

Target `y` adalah `pm25_target`, yaitu indeks PM2.5 aktual pada tanggal T.

Sampel dikeluarkan bila target kosong atau salah satu fitur riwayat tidak lengkap. Hasilnya **3.081 sampel siap model** dari 3.500 hari-stasiun:

- 60 sampel dibuang karena target PM2.5 kosong.
- 359 sampel dibuang karena riwayat `lag_1`, `lag_3`, `lag_7`, atau `rolling_mean_7` tidak lengkap.
- Tidak ada imputasi nilai historis, sehingga tidak ada informasi masa depan yang bocor ke fitur.

### 3.3 Split temporal

| Split | Tanggal | Baris | Keterangan |
|---|---|---:|---|
| Train | 1 Jan 2024 – 4 Mei 2025 | 2.186 | 490 tanggal dialokasikan; 483 tanggal memiliki sampel karena 7 hari pertama belum punya riwayat |
| Validation | 5 Mei – 17 Agu 2025 | 446 | 105 tanggal |
| Test | 18 Agu – 30 Nov 2025 | 449 | 105 tanggal |

Tidak ada tanggal yang muncul di lebih dari satu split. Test set tidak dipakai untuk memilih hyperparameter.

### 3.4 Model

**Naive Forecast (baseline):**

```text
prediksi_T = nilai_pm25 pada T−1
```

Tidak ada pelatihan atau model tersimpan untuk baseline.

**XGBoost Regressor:**

- Objective `reg:squarederror`, `tree_method="hist"`.
- Pencarian hyperparameter pada validation set: 18 kombinasi.
  - `max_depth`: 2, 3, 4
  - `learning_rate`: 0,03; 0,05; 0,10
  - `min_child_weight`: 1, 5
  - `subsample` = 0,8; `colsample_bytree` = 0,8; `reg_lambda` = 1,0
- Maksimum 1.500 boosting rounds dengan early stopping 50 rounds; metrik early stopping = MAE.
- Konfigurasi terbaik: `max_depth=2`, `learning_rate=0,10`, `min_child_weight=5`, **75 rounds** setelah refit.
- Pemilihan model memakai **validation MAE**, lalu model final dilatih ulang pada train+validation.
- Pipeline menyimpan encoder stasiun dan model dalam satu artefak: `models/xgboost_model.joblib`.

### 3.5 Evaluasi

- **MAE:** rata-rata `|actual − prediction|`.
- **RMSE:** akar dari rata-rata `(actual − prediction)²`.
- Keduanya dalam satuan poin indeks PM2.5.
- Naive dan XGBoost dievaluasi pada DataFrame test yang identik.
- Prediksi XGBoost dijaga pada rentang fisis indeks 0–500; pada hasil ini tidak ada prediksi yang terpotong karena nilai minimum prediksi sekitar 25,4 dan maksimum sekitar 96,5.

---

## 4. Struktur Proyek

```text
PEMATIK 2026/
├── data/
│   ├── ispu_raw.xls                   # Dataset mentah (HTML table)
│   └── ispu_processed.csv             # Dihasilkan preprocessing.py
├── src/
│   ├── config.py                      # Path, fitur, split, parameter bersama
│   ├── preprocessing.py               # Cleaning + kalender + fitur + split
│   ├── train_model.py                 # Tuning validation + model final
│   └── evaluate.py                    # Naive + XGBoost, metrik, grafik
├── notebooks/
│   └── analysis.ipynb                 # EDA lengkap dengan output
├── models/
│   └── xgboost_model.joblib           # Pipeline encoder + model
├── outputs/
│   ├── predictions.csv                # Prediksi test per tanggal/stasiun
│   ├── metrics.csv                    # MAE/RMSE test
│   ├── metrics_per_station.csv        # Metrik per stasiun
│   ├── tuning_results.csv             # 18 kandidat hyperparameter
│   ├── training_summary.json          # Ringkasan training + versi library
│   ├── preprocessing_log.json         # Audit cleaning dan split
│   ├── run_summary.json               # Ringkasan evaluasi test
│   └── figures/
│       ├── actual_vs_predicted.png    # Wajib
│       ├── model_comparison.png       # Wajib
│       ├── feature_importance.png     # Pendukung
│       └── eda_*.png                  # Pendukung dari notebook
├── report/                            # Reservasi technical report
├── presentation/                      # Reservasi presentasi
├── run_pipeline.ps1                   # Menjalankan tiga script berurutan
├── requirements.txt
└── README.md
```

> **Catatan Git:** `.gitignore` mengecualikan `data/ispu_raw.xls` dan `models/*.joblib` dari repositori. Kedua file tetap tersedia secara lokal dan perlu disertakan saat pengumpulan karya sesuai ketentuan panitia.

---

## 5. Cara Menjalankan

### 5.1 Prasyarat

- Python 3.11 atau 3.12 (proyek ini diuji pada Python 3.11.9).
- Windows PowerShell (langkah Command Prompt juga disertakan).

### 5.2 Membuat virtual environment

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

Jika aktivasi PowerShell dibatasi:

```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\.venv\Scripts\Activate.ps1
```

Atau gunakan Command Prompt:

```bat
.venv\Scripts\activate.bat
```

### 5.3 Instal dependensi

```powershell
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Dependensi utama: `pandas`, `numpy`, `scikit-learn`, `xgboost`, `matplotlib`, `joblib`, `jupyterlab`, dan `lxml` untuk parsing HTML.

### 5.4 Menjalankan pipeline utama

Dari folder root proyek:

```powershell
python src/preprocessing.py
python src/train_model.py
python src/evaluate.py
```

Atau satu perintah:

```powershell
.\run_pipeline.ps1
```

Jika eksekusi script PowerShell diblokir kebijakan sistem:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

Urutan ini penting. Jika satu tahap gagal, perbaiki dulu sebelum lanjut.

### 5.5 Menjalankan EDA

```powershell
jupyter lab
```

Buka `notebooks/analysis.ipynb`, lalu **Run All**. Notebook sudah berisi output eksekusi, tetapi dapat dijalankan ulang untuk memverifikasi.

### 5.6 Output yang dihasilkan

| Tahap | Output |
|---|---|
| `preprocessing.py` | `data/ispu_processed.csv`, `outputs/preprocessing_log.json` |
| `train_model.py` | `models/xgboost_model.joblib`, `outputs/tuning_results.csv`, `outputs/training_summary.json` |
| `evaluate.py` | `outputs/predictions.csv`, `outputs/metrics.csv`, `outputs/metrics_per_station.csv`, `outputs/run_summary.json`, tiga grafik di `outputs/figures/` |

---

## 6. Catatan Teknis dan Keterbatasan

- **Skenario one-step-ahead.** Fitur `lag_1` memakai nilai aktual T−1, sehingga model mengasumsikan observasi harian tersedia sebelum prediksi dibuat. Untuk peramalan beberapa hari ke depan tanpa pembaruan observasi, skenario dan fitur harus diubah.
- **Hanya dua metode dan dua metrik** sesuai batasan eksperimen. Tidak ada model pembanding lain di pipeline utama.
- **Fitur terbatas.** Model hanya memakai riwayat PM2.5, stasiun, dan bulan. Faktor meteorologi, lalu lintas, dan sumber emisi tidak dimasukkan.
- **Missing value tidak diimputasi.** Sampel yang riwayatnya tidak lengkap dikeluarkan. Bila pola missing tidak acak, hal ini dapat memengaruhi estimasi performa.
- **Satu validation window.** Hyperparameter dipilih dari 18 kandidat pada satu periode validation kronologis, bukan cross-validation bergulir. Hasil dapat berbeda bila batas split diubah.
- **Hasil spesifik dataset.** Angka MAE/RMSE berlaku untuk periode dan stasiun pada dataset ini; bukan klaim performa umum untuk kualitas udara Jakarta.
- **Model final memakai train+validation.** Keputusan ini ditetapkan sebelum test dievaluasi. Validation metrics yang dilaporkan berasal dari model hasil tuning yang dilatih pada train saja; test metrics berasal dari model final yang dilatih pada train+validation.
- **Angka di README** konsisten dengan `outputs/metrics.csv`, `outputs/metrics_per_station.csv`, `outputs/run_summary.json`, dan `outputs/training_summary.json`.

---

## 7. Checklist Reproduksibilitas

- [x] Dataset mentah tersimpan di `data/ispu_raw.xls`.
- [x] Tanggal tidak valid tercatat di `outputs/preprocessing_log.json`.
- [x] Lima stasiun distandardisasi; lag/rolling dihitung per stasiun.
- [x] Lag dan rolling hanya memakai data sebelum tanggal target.
- [x] Train/validation/test dipisah berdasarkan tanggal, tanpa tumpang tindih.
- [x] Naive dan XGBoost dievaluasi pada baris test yang sama.
- [x] `predictions.csv`, `metrics.csv`, dan dua grafik wajib berhasil dihasilkan.
- [x] Hyperparameter dipilih hanya dari validation; test tidak dipakai untuk tuning.
- [x] Versi library dan parameter model tercatat di `outputs/training_summary.json`.
- [x] Notebook EDA dapat dijalankan ulang dari root proyek.

---

## 8. Versi yang Diuji

| Library | Versi |
|---|---|
| Python | 3.11.9 |
| pandas | 2.2.3 |
| numpy | 1.26.4 |
| scikit-learn | 1.5.0 |
| xgboost | 2.1.4 |
| matplotlib | 3.9.0 |
| joblib | 1.4.2 |
| lxml | 6.0.2 |
