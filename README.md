# Prediksi Nilai Indeks PM2.5 Harian DKI Jakarta

Prediksi indeks PM2.5 untuk hari berikutnya di lima stasiun pemantau ISPU DKI
Jakarta. Naive Forecast dipakai sebagai baseline, XGBoost sebagai metode utama.

Judul karya:

> Prediksi Nilai Indeks PM2.5 Harian Berdasarkan Data Indeks Standar Pencemar Udara
> (ISPU) di DKI Jakarta Menggunakan Metode Naïve Forecast dan XGBoost

Tim: zian disuruh revisi skripsi malah daftar lomba data mining

## Hasil

Pengujian dilakukan pada 449 observasi test, yaitu 105 tanggal antara 18 Agustus
sampai 30 November 2025.

| Model | MAE | RMSE |
|---|---:|---:|
| Naive Forecast | 11,9978 | 15,7731 |
| XGBoost | 10,5579 | 13,5799 |

Satuannya poin indeks PM2.5, jadi makin kecil makin bagus. XGBoost menurunkan MAE
12,0% dan RMSE 13,9% dibanding Naive Forecast, dan hasilnya lebih bagus di kelima
stasiun.

Saat hyperparameter masih dipilih di validation set, MAE Naive Forecast 15,0157 dan
MAE XGBoost 12,4926. Validation tidak dipakai untuk menilai hasil akhir.

![Aktual vs prediksi](outputs/figures/actual_vs_predicted.png)

![Perbandingan MAE dan RMSE](outputs/figures/model_comparison.png)

![Kontribusi fitur](outputs/figures/feature_importance.png)

## Data

Sumbernya dataset ISPU DKI Jakarta dari Portal Data Terpadu DKI Jakarta, file
`data/ispu_raw.xls`. Ekstensinya `.xls`, tapi isinya tabel HTML, jadi dibaca
dengan `pandas.read_html()` bukan `read_excel()`.

Ukuran data mentah 3.501 baris dan 13 kolom, mencakup 700 tanggal sejak 1 Januari
2024 sampai 30 November 2025 di lima stasiun: DKI1 Bundaran HI, DKI2 Kelapa
Gading, DKI3 Jagakarsa, DKI4 Lubang Buaya, dan DKI5 Kebon Jeruk.

Target yang diprediksi adalah `pm_duakomalima`, yaitu nilai indeks PM2.5, bukan
konsentrasi dalam µg/m³.

Tiga hal yang perlu dibersihkan. Pertama, 61 nilai PM2.5 kosong dan tidak diimputasi.
Kedua, satu baris bertanggal 31 September 2025 di DKI5 dibuang karena tanggalnya
tidak ada di kalender; baris ini tercatat di `outputs/preprocessing_log.json`.
Ketiga, nama stasiun ditulis berbeda-beda dalam file mentah, misalnya "DKI1
BUNDARAN HOTEL INDONESIA (HI)", "DKI1 BUNDERAN HI", dan "DKI1 BUNDARAN HOTEL
INDONESIA HI". Semuanya dinormalkan jadi lima nama baku.

Setelah fitur riwayat terbentuk, tersisa 3.081 sampel siap model: 2.186 baris train,
446 baris validation, dan 449 baris test.

## Metode

Fitur yang dipakai ada enam: `lag_1`, `lag_3`, `lag_7`, `rolling_mean_7`, `bulan`,
dan `stasiun` yang di-encode one-hot. Arti lag_1 adalah nilai PM2.5 pada hari
sebelumnya untuk stasiun yang sama, jadi untuk target tanggal T, lag_1 adalah nilai
T-1. Semua lag dan rolling dihitung per stasiun dari nilai sebelum tanggal target.
PM10 ada di data mentah dan diperiksa kualitasnya, tapi tidak dipakai sebagai fitur.

Pembagian data memakai tanggal unik, bukan pengocokan baris acak, dengan rasio 70%
train, 15% validation, 15% test. Tidak ada tanggal yang masuk dua bagian. Pembagian
dilakukan sebelum fitur dibuat, jadi tidak ada kebocoran dari data test.

Naive Forecast memakai nilai T-1 langsung sebagai prediksi T. XGBoost diuji dengan
18 kombinasi hyperparameter dan dipilih berdasarkan MAE validation. Konfigurasi
terpilih adalah `max_depth=2`, `learning_rate=0,10`, `min_child_weight=5` dengan 75
boosting rounds. Setelah itu model dilatih ulang pada train+validation, lalu diuji
sekali pada test set.

Metrik evaluasi hanya MAE dan RMSE, keduanya dalam poin indeks PM2.5.

## Struktur folder

```text
.
├── report/
│   └── technical_report.pdf
├── src/
│   ├── preprocessing.py
│   ├── train_model.py
│   └── evaluate.py
├── notebooks/
│   └── analysis.ipynb
├── outputs/
│   ├── predictions.csv
│   └── figures/
├── presentation/
│   └── presentasi.pptx
└── README.md
```

Folder `data/` dan `models/` ada di luar daftar di atas karena keduanya dibutuhkan
untuk menjalankan ulang pipeline. `data/` berisi file mentah dan hasil
preprocessing, `models/` berisi model XGBoost yang sudah dilatih. Keduanya bisa
dibuat ulang dari nol dengan tiga script di `src/`.

## Cara menjalankan

Semua perintah dijalankan dari folder root proyek.

Buat virtual environment dan pasang dependensi dulu:

```powershell
python --version
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Kalau aktivasi diblokir kebijakan PowerShell, jalankan `Set-ExecutionPolicy -Scope
Process Bypass` lebih dulu.

Setelah itu jalankan tiga script secara berurutan:

```powershell
python src\preprocessing.py
python src\train_model.py
python src\evaluate.py
```

`preprocessing.py` membaca data mentah, membuang tanggal tidak valid, membuat
kalender harian, menghitung fitur lag dan rolling, lalu membagi data. Di layar
akan terlihat 3.501 baris mentah, 61 PM2.5 kosong, 3.081 sampel siap model, dan
pembagian 2.186/446/449 baris.

`train_model.py` menguji 18 kandidat pada validation set, memilih yang MAE-nya
terkecil, lalu melatih ulang model final pada train+validation. Konfigurasi terbaik
yang akan tercetak adalah `depth=2 lr=0.10 mcw=5 rounds=75`.

`evaluate.py` memprediksi test set dengan Naive dan XGBoost, menghitung MAE dan
RMSE, lalu menyimpan berkas keluaran dan tiga grafik.

Ketiganya bisa dijalankan sekaligus:

```powershell
.\run_pipeline.ps1
```

Kalau eksekusi file PowerShell diblokir:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_pipeline.ps1
```

Urutan preprocessing, training, lalu evaluasi tidak boleh dibalik karena tiap
script membaca output script sebelumnya.

Notebook `notebooks/analysis.ipynb` terpisah dari pipeline. Isinya analisis data
eksploratif: struktur file, kualitas data, kalender, tren harian, pola bulanan,
korelasi fitur, dan pemeriksaan hasil pembagian data. Notebook memakai fungsi dari
`src/preprocessing.py` supaya hasilnya sama dengan pipeline utama.

```powershell
jupyter lab
```

Buka `notebooks/analysis.ipynb` lalu Run All. Notebook sudah berisi output, tapi
boleh dijalankan ulang untuk memeriksa.

## Catatan

Model ini hanya berlaku untuk prediksi satu hari ke depan dengan observasi harian
yang tersedia. Untuk beberapa hari ke depan, fitur dan skenario harus diubah.

Nilai kosong tidak diimputasi. Sampel dengan target kosong atau riwayat lag dan
rolling tidak lengkap dikeluarkan, jadi hasilnya bisa bias kalau pola kosongnya
tidak acak.

Hyperparameter dipilih dari satu periode validation, bukan cross-validation
ber gulir. Hasilnya bisa berbeda kalau batas split diubah.

Angka MAE dan RMSE di sini berlaku untuk dataset dan periode ini, bukan klaim umum
tentang kualitas udara Jakarta.

`.gitignore` mengecualikan `data/ispu_raw.xls` dan `models/*.joblib`. Keduanya ada
di folder lokal tapi perlu diunggah manual saat repository dikirim ke juri.

Dipakai dengan Python 3.11.9, pandas 2.2.3, scikit-learn 1.5.0, dan xgboost 2.1.4.
