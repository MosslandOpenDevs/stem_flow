$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
$python = Join-Path $project ".venv\Scripts\python.exe"

if (-not (Test-Path -LiteralPath $python)) {
  throw "Run scripts\setup-local.ps1 first."
}

if (-not $env:STEMFLOW_STORAGE_ROOT) {
  $env:STEMFLOW_STORAGE_ROOT = Join-Path $project "storage"
}

# Deliberately no tool paths here. Setting them would take priority over
# everything else and silently shadow assets vendored into the project. Configure
# them in .env, or let worker/separate_v4.py discover them.

Set-Location $project
& $python -m uvicorn server.app:app --host 127.0.0.1 --port 8000
