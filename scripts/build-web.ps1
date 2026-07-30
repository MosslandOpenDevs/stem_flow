$ErrorActionPreference = "Stop"
. (Join-Path $PSScriptRoot "common.ps1")

$node = Resolve-StemFlowNode
Set-Location $StemFlowProject
& $node ".\node_modules\vinext\dist\cli.js" build
