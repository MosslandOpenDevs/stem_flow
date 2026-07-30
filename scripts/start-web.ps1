$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot
if (-not $env:NEXT_PUBLIC_STEMFLOW_API_BASE) {
  $env:NEXT_PUBLIC_STEMFLOW_API_BASE = "http://127.0.0.1:8000"
}
# STEMFLOW_NODE overrides discovery when the Node on PATH is too old.
$node = $env:STEMFLOW_NODE
if ([string]::IsNullOrWhiteSpace($node) -or -not (Test-Path -LiteralPath $node)) {
  $node = (Get-Command node -ErrorAction SilentlyContinue).Source
}
$nodeMajor = if ($node) { [int]((& $node --version).TrimStart("v").Split(".")[0]) } else { 0 }
if ($nodeMajor -lt 22) {
  $found = if ($node) { (& $node --version) } else { "none" }
  throw "StemFlow requires Node 22.13 or newer (found $found). Install it, or set STEMFLOW_NODE to a newer node.exe."
}
Set-Location $project
& $node ".\node_modules\vinext\dist\cli.js" dev
