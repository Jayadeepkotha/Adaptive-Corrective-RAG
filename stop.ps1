Write-Host "========================================================" -ForegroundColor Yellow
Write-Host "       Stopping Confidence-RAG and Services" -ForegroundColor Yellow
Write-Host "========================================================" -ForegroundColor Yellow
Write-Host ""

# 1. Stop Port 8000 process
Write-Host "[1/2] Terminating Web Server on port 8000..." -ForegroundColor Gray
$connections = Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue
foreach ($conn in $connections) {
    try {
        Stop-Process -Id $conn.OwningProcess -Force -ErrorAction SilentlyContinue
        Write-Host "  -> Terminated process on port 8000 (PID: $($conn.OwningProcess))" -ForegroundColor Green
    } catch {}
}

# 2. Stop Ollama
Write-Host "[2/2] Stopping Ollama background services..." -ForegroundColor Gray
Stop-Process -Name "ollama", "ollama app" -Force -ErrorAction SilentlyContinue
Write-Host "  -> Ollama stopped." -ForegroundColor Green

Write-Host ""
Write-Host "========================================================" -ForegroundColor Green
Write-Host "  [OK] All Confidence-RAG & Ollama services are OFF." -ForegroundColor Green
Write-Host "========================================================" -ForegroundColor Green
Write-Host ""
