# Prediksi Indeks PM2.5 Harian di DKI Jakarta

Proyek ini membandingkan Naive Forecast dan XGBoost untuk memprediksi indeks PM2.5
pada hari berikutnya di lima stasiun pemantauan ISPU DKI Jakarta.

Evaluasi memakai skenario one-step-ahead. Saat memprediksi tanggal T, nilai aktual
sampai T-1 boleh dipakai sebagai fitur, tetapi nilai T dan sesudahnya tidak. Ini
bukan peramalan banyak hari ke depan tanpa pembaruan observasi.

## Hasil

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Forecast | 11,9978 | 15,7731 |
| XGBoost | 10,5579 | 13,5799 |

Diukur pada 449 observasi test (105 tanggal, 18 Agustus s.d. 30 November 2025).
Satuannya poin indeks PM2.5, jadi semakin kecil semakin baik. XGBoost menurunkan
MAE 12,0% dan RMSE 13,9% dibanding Naive Forecast, dan lebih rendah di kelima
stasiun.

Pada validation set, yang dipakai untuk memilih hyperparameter, MAE Naive 15,0157
dan MAE XGBoost 12,4926. Validation tidak dipakai untuk menilai hasil akhir.

![Aktual vs prediksi](outputs/figures/actual_vs_predicted.png)

![Perbandingan MAE dan RMSE](outputs/figures/model_comparison.png)

![Kontribusi fitur](outputs/figures/feature_importance.png)

## Data

- Sumber: dataset ISPU DKI Jakarta, file `data/ispu_raw.xls`. Isinya tabel HTML
  walau ekstensinya `.xls`, jadi dibaca dengan `pandas.read_html()`.
- Ukuran: 3.501 baris, 13 kolom, 700 tanggal (1 Januari 2024 s.d. 30 November
  2025), dan 5 stasiun.
- Target: `pm_duakomalima`, yaitu indeks PM2.5, bukan konsentrasi µg/m³.
- Ada 61 nilai PM2.5 kosong. Nilai kosong tidak diimputasi.
- Satu baris bertanggal 31 September 2025 (DKI5) dibuang karena tanggalnya tidak
  ada. Baris tersebut tercatat di `outputs/preprocessing_log.json`.
- Setelah fitur riwayat terbentuk, tersisa 3.081 sampel siap model: train 2.186
  baris, validation 446 baris, dan test 449 baris.

## Metode

Fitur yang dipakai adalah `lag_1`, `lag_3`, `lag_7`, `rolling_mean_7`, `bulan`,
dan `stasiun` (one-hot). Lag dan rolling dihitung per stasiun dari nilai sebelum
tanggal target.

Split dilakukan pada tanggal unik, bukan baris acak: 70% train, 15% validation,
15% test. Tidak ada tanggal yang masuk dua split.

Naive Forecast memakai nilai T-1 sebagai prediksi T. XGBoost diuji dengan 18
kombinasi hyperparameter pada validation MAE. Konfigurasi terpilih adalah
`max_depth=2`, `learning_rate=0,10`, dan `min_child_weight=5` dengan 75 boosting
rounds. Setelah itu model dilatih ulang pada train+validation dan diuji satu kali
pada test.

Metrik evaluasi hanya MAE dan RMSE, keduanya dalam poin indeks PM2.5.

## Struktur

```text
.
├── data/
│   ├── ispu_raw.xls
│   └── ispu_processed.csv
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── models/
│   └── xgboost_model.joblib
├── outputs/
│   ├── predictions.csv
│   ├── metrics.csv
│   ├── metrics_per_station.csv
│   ├── tuning_results.csv
│   ├── preprocessing_log.json
│   ├── training_summary.json
│   ├── run_summary.json
│   └── figures/
│       ├── actual_vs_predicted.png
│       ├── model_comparison.png
│       └── feature_importance.png
├── report/
├── presentation/
├── run_pipeline.ps1
├── requirements.txt
└── README.md
```

## Cara menjalankan

Jalankan dari folder root proyek.

1. Cek Python dan buat virtual environment.

   ```powershell
   python --version
   python -m venv .venv
   .\.venv\Scripts\Activate.ps1
   ```

   Kalau aktivasi diblokir kebijakan PowerShell:

   ```powershell
   Set-ExecutionPolicy -Scope Process Bypass
   .\.venv\Scripts\Activate.ps1
   ```

2. Instal dependensi. Butuh koneksi internet pada langkah ini.

   ```powershell
   python -m pip install --upgrade pip
   pip install -r requirements.txt
   ```

3. Jalankan preprocessing.

   ```powershell
   python src/preprocessing.py
   ```

   Proses ini membaca data mentah, membuang tanggal tidak valid, membuat kalender
   harian, menghitung lag/rolling, lalu membagi tanggal. Hasilnya
   `data/ispu_processed.csv` dan `outputs/preprocessing_log.json`. Di layar akan
   terlihat 3.501 baris mentah, 61 PM2.5 kosong, 3.081 sampel siap model, dan
   pembagian 2.186/446/449 baris.

4. Jalankan training.

   ```powershell
   python src/train_model.py
   ```

   Script menguji 18 kandidat pada validation set, memilih yang MAE-nya terkecil,
   lalu melatih ulang model final pada train+validation. Hasilnya
   `models/xgboost_model.joblib`, `outputs/tuning_results.csv`, dan
   `outputs/training_summary.json`. Konfigurasi terbaik yang akan tercetak adalah
   `depth=2 lr=0.10 mcw=5 rounds=75`.

5. Jalankan evaluasi.

   ```powershell
   python src/evaluate.py
   ```

   Script memprediksi test set dengan Naive dan XGBoost, menghitung MAE/RMSE, lalu
   menyimpan `outputs/predictions.csv`, `outputs/metrics.csv`,
   `outputs/metrics_per_station.csv`, `outputs/run_summary.json`, dan tiga grafik
   di `outputs/figures/`.

6. Untuk EDA, buka notebook.

   ```powershell
   jupyter lab
   ```

   Buka `notebooks/analysis.ipynb`, lalu Run All. Notebook sudah berisi output,
   tetapi bisa dijalankan ulang untuk memeriksa hasilnya.

Ketiga script juga bisa dijalankan sekaligus:

```powershell
.\run_pipeline.ps1
```

Kalau eksekusi file PowerShell diblokir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

Urutan preprocessing, training, lalu evaluasi tidak boleh dibalik karena tiap
script membaca output script sebelumnya.

## Catatan

- Model hanya valid untuk prediksi satu hari ke depan dengan observasi harian
  yang tersedia. Untuk beberapa hari ke depan, fitur dan skenario harus diubah.
- Nilai yang kosong tidak diimputasi. Sampel dengan riwayat tidak lengkap
  dikeluarkan, sehingga hasil bisa bias jika pola kosongnya tidak acak.
- Hyperparameter dipilih dari satu periode validation, bukan cross-validation
  bergulir. Hasil bisa berbeda kalau batas split diubah.
- Angka MAE/RMSE berlaku untuk dataset dan periode ini, bukan klaim umum tentang
  kualitas udara Jakarta.
- `.gitignore` mengecualikan `data/ispu_raw.xls` dan `models/*.joblib`. Keduanya
  tetap ada di folder lokal dan perlu disertakan saat pengumpulan karya.
- Diuji dengan Python 3.11.9, pandas 2.2.3, scikit-learn 1.5.0, dan xgboost 2.1.4.
