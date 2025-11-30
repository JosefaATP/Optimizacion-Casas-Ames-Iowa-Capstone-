# Script para ejecutar benchmark de 100 casas en background
# Uso: .\run_benchmark_100.ps1

$workdir = "c:\Users\valen\OneDrive - uc.cl\UNIVERSIDAD\2024-1\Bases de datos\Optimizacion-Casas-Ames-Iowa-Capstone-"
$outdir = "bench_final_100"
$logfile = "bench_final_100.log"

cd $workdir

# Limpiar directorio anterior si existe
Remove-Item $outdir -Recurse -ErrorAction SilentlyContinue

Write-Host "=== INICIANDO BENCHMARK DE 100 CASAS ===" -ForegroundColor Cyan
Write-Host "Salida: $outdir" -ForegroundColor Cyan
Write-Host "Log: $logfile" -ForegroundColor Cyan
Write-Host "Timestamp: $(Get-Date)" -ForegroundColor Cyan
Write-Host ""

# Ejecutar benchmark
$startTime = Get-Date
&".venv311\Scripts\python.exe" -m optimization.remodel.benchmark_remodel `
    --basecsv "data/processed/base_completa_sin_nulos.csv" `
    --n_houses 100 `
    --seed 42 `
    --outdir $outdir `
    2>&1 | Tee-Object -FilePath $logfile

$endTime = Get-Date
$duration = $endTime - $startTime

Write-Host ""
Write-Host "=== BENCHMARK COMPLETADO ===" -ForegroundColor Green
Write-Host "Duración total: $($duration.TotalMinutes.ToString('F1')) minutos" -ForegroundColor Green
Write-Host "Hora de finalización: $(Get-Date)" -ForegroundColor Green
Write-Host ""
Write-Host "Archivos generados:" -ForegroundColor Cyan
Get-ChildItem $outdir -File | Select-Object Name, @{Name="Size (KB)"; Expression={[math]::Round($_.Length/1KB, 2)}}
