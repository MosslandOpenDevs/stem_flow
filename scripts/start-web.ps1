$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

# vinext loads .env itself, but only for keys absent from the environment. Going
# through Get-Setting keeps this fallback from shadowing a value set there.
$apiBase = Get-Setting "NEXT_PUBLIC_STEMFLOW_API_BASE"
if ([string]::IsNullOrWhiteSpace($apiBase)) { $apiBase = "http://127.0.0.1:8000" }
$env:NEXT_PUBLIC_STEMFLOW_API_BASE = $apiBase

$node = Resolve-StemFlowNode
Set-Location $StemFlowProject
& $node ".\node_modules\vinext\dist\cli.js" dev
