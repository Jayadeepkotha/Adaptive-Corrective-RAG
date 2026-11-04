Write-Host "========================================================" -ForegroundColor Cyan
Write-Host "       Starting Confidence-RAG Adaptive System" -ForegroundColor Cyan
Write-Host "========================================================" -ForegroundColor Cyan
Write-Host ""

# 1. Check & Start Ollama Local Safety Net
$ollamaProc = Get-Process -Name "ollama" -ErrorAction SilentlyContinue
if (-not $ollamaProc) {
    Write-Host "[1/2] Starting Ollama server in background..." -ForegroundColor Yellow
    Start-Process -FilePath "ollama" -ArgumentList "serve" -WindowStyle Hidden
    Start-Sleep -Seconds 2
} else {
    Write-Host "[1/2] Ollama is already active on http://127.0.0.1:11434" -ForegroundColor Green
}

# 2. Clear any lingering process on port 8000 so there's never an "Address in use" error
$port8000 = Get-NetTCPConnection -LocalPort 8000 -State Listen -ErrorAction SilentlyContinue
if ($port8000) {
    Write-Host "[INFO] Freeing previous process on port 8000..." -ForegroundColor Yellow
    Stop-Process -Id $port8000.OwningProcess -Force -ErrorAction SilentlyContinue
    Start-Sleep -Seconds 1
}

# 3. Launch Web Server & Open Browser
Write-Host "[2/2] Starting Web Server on http://localhost:8000..." -ForegroundColor Green
Start-Process "http://localhost:8000"
Write-Host ""
Write-Host "Confidence-RAG is running at: http://localhost:8000" -ForegroundColor Cyan
Write-Host "Press Ctrl+C to stop the server anytime." -ForegroundColor Gray
Write-Host ""

python -m uvicorn api:app --host 0.0.0.0 --port 8000
