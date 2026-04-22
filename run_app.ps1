Param(
  [int]$Port = 8501,
  [string]$Addr = "localhost",
  [switch]$OpenBrowser
)

$ErrorActionPreference = "Stop"

$python = Join-Path $PSScriptRoot ".venv\\Scripts\\python.exe"
if (-not (Test-Path $python)) {
  Write-Host "找不到 .venv，請先建立虛擬環境並安裝套件：" -ForegroundColor Yellow
  Write-Host "  .\\scripts\\setup_venv.ps1" -ForegroundColor Yellow
  exit 1
}

$env:STREAMLIT_PORT = "$Port"
$env:STREAMLIT_ADDR = "$Addr"
if ($OpenBrowser) { $env:STREAMLIT_OPEN_BROWSER = "1" }

Write-Host "啟動指令：" -ForegroundColor Cyan
Write-Host "  $python .\\main.py" -ForegroundColor Cyan
Write-Host ""

& $python ".\\main.py"

