# Menjalankan preprocessing, training, dan evaluasi secara berurutan.
# Penggunaan: .\run_pipeline.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "Preprocessing..." -ForegroundColor Cyan
python src/preprocessing.py
if ($LASTEXITCODE -ne 0) { throw "Preprocessing gagal. Pipeline dihentikan." }

Write-Host "Training XGBoost..." -ForegroundColor Cyan
python src/train_model.py
if ($LASTEXITCODE -ne 0) { throw "Training gagal. Pipeline dihentikan." }

Write-Host "Evaluasi..." -ForegroundColor Cyan
python src/evaluate.py
if ($LASTEXITCODE -ne 0) { throw "Evaluasi gagal. Pipeline dihentikan." }

Write-Host "Selesai. Hasil ada di outputs/." -ForegroundColor Green
