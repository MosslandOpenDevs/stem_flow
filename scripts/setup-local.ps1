$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$venv = Join-Path $project ".venv"

& (Join-Path $PSScriptRoot "preflight.ps1")

if (-not (Test-Path -LiteralPath (Join-Path $venv "Scripts\python.exe"))) {
  python -m venv $venv
}

$python = Join-Path $venv "Scripts\python.exe"
& $python -m pip install --upgrade pip
& $python -m pip install -r (Join-Path $project "server\requirements.txt")

Push-Location $project
try {
  npm install
} finally {
  Pop-Location
}

Write-Output "StemFlow local environment is ready."
