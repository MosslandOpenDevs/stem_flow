$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
if (-not $env:NEXT_PUBLIC_STEMFLOW_API_BASE) {
  $env:NEXT_PUBLIC_STEMFLOW_API_BASE = "http://127.0.0.1:8000"
}
$node = (Get-Command node -ErrorAction SilentlyContinue).Source
$nodeMajor = if ($node) { [int]((& $node --version).TrimStart("v").Split(".")[0]) } else { 0 }
if ($nodeMajor -lt 22) {
  $bundledNode = Join-Path $env:LOCALAPPDATA "Temp\node-v22.17.0-win-x64\node.exe"
  if (Test-Path -LiteralPath $bundledNode) {
    $node = $bundledNode
  } else {
    throw "StemFlow requires Node 22.13 or newer."
  }
}
Set-Location $project
& $node ".\node_modules\vinext\dist\cli.js" dev
