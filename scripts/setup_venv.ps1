Param(
  [string]$VenvDir = ".venv"
)

$ErrorActionPreference = "Stop"

Write-Host "Create venv: $VenvDir"
python -m venv $VenvDir

$python = Join-Path $VenvDir "Scripts\\python.exe"
if (-not (Test-Path $python)) {
  throw "Cannot find venv python at $python"
}

Write-Host "Upgrade pip"
& $python -m pip install --upgrade pip

Write-Host "Install requirements.txt"
& $python -m pip install -r "requirements.txt"

Write-Host ""
Write-Host "Done."
Write-Host "Activate:"
Write-Host "  $VenvDir\\Scripts\\Activate.ps1"
Write-Host "Run app:"
Write-Host "  python -m streamlit run main.py"

