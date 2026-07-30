# Shared configuration helpers for the StemFlow PowerShell scripts.
#
# Dot-source this instead of duplicating the logic, so every script resolves
# settings the same way the Python side does and no script silently ignores
# a value the user put in .env.

$StemFlowProject = Split-Path -Parent $PSScriptRoot

# Read .env the same way server/config.py does, so these scripts never act on a
# resolution that differs from what actually runs. Real environment variables
# still win, matching load_dotenv(override=False).
$StemFlowDotEnv = @{}
$stemFlowEnvPath = Join-Path $StemFlowProject ".env"
if (Test-Path -LiteralPath $stemFlowEnvPath) {
  foreach ($line in Get-Content -LiteralPath $stemFlowEnvPath) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    $split = $trimmed.IndexOf("=")
    if ($split -lt 1) { continue }
    $key = $trimmed.Substring(0, $split).Trim()
    $value = $trimmed.Substring($split + 1).Trim().Trim('"').Trim("'")
    if ($value -ne "") { $StemFlowDotEnv[$key] = $value }
  }
}

function Get-Setting($name) {
  $value = [Environment]::GetEnvironmentVariable($name)
  if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
  if ($StemFlowDotEnv.ContainsKey($name)) { return $StemFlowDotEnv[$name] }
  return $null
}

# The web app needs Node 22.13+. STEMFLOW_NODE overrides discovery when the Node
# on PATH is older; it goes through Get-Setting so setting it in .env works,
# which is how both README.md and .env.example document it.
function Resolve-StemFlowNode {
  $node = Get-Setting "STEMFLOW_NODE"
  if ([string]::IsNullOrWhiteSpace($node) -or -not (Test-Path -LiteralPath $node)) {
    $node = (Get-Command node -ErrorAction SilentlyContinue).Source
  }
  $major = if ($node) { [int]((& $node --version).TrimStart("v").Split(".")[0]) } else { 0 }
  if ($major -lt 22) {
    $found = if ($node) { (& $node --version) } else { "none" }
    throw "StemFlow requires Node 22.13 or newer (found $found). Install it, or set STEMFLOW_NODE to a newer node.exe."
  }
  return $node
}
