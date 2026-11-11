# Menjalankan seluruh pipeline utama dari folder root.
# Penggunaan: .\run_pipeline.ps1

$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $projectRoot

Write-Host "=== 1/3 Preprocessing ===" -ForegroundColor Cyan
python src/preprocessing.py
if ($LASTEXITCODE -ne 0) { throw "Preprocessing gagal. Pipeline dihentikan." }

Write-Host "=== 2/3 Training XGBoost ===" -ForegroundColor Cyan
python src/train_model.py
if ($LASTEXITCODE -ne 0) { throw "Training gagal. Pipeline dihentikan." }

Write-Host "=== 3/3 Evaluasi ===" -ForegroundColor Cyan
python src/evaluate.py
if ($LASTEXITCODE -ne 0) { throw "Evaluasi gagal. Pipeline dihentikan." }

Write-Host ""
Write-Host "Selesai. Lihat outputs/metrics.csv dan outputs/figures/." -ForegroundColor Green
