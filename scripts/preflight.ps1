$ErrorActionPreference = "Stop"
$project = Split-Path -Parent $PSScriptRoot

# Read .env the same way server/config.py does, so this script never reports a
# resolution that differs from what actually runs. Real environment variables
# still win, matching load_dotenv(override=False).
$dotenv = @{}
$envPath = Join-Path $project ".env"
if (Test-Path -LiteralPath $envPath) {
  foreach ($line in Get-Content -LiteralPath $envPath) {
    $trimmed = $line.Trim()
    if ($trimmed -eq "" -or $trimmed.StartsWith("#")) { continue }
    $split = $trimmed.IndexOf("=")
    if ($split -lt 1) { continue }
    $key = $trimmed.Substring(0, $split).Trim()
    $value = $trimmed.Substring($split + 1).Trim().Trim('"').Trim("'")
    if ($value -ne "") { $dotenv[$key] = $value }
  }
}

function Get-Setting($name) {
  $value = [Environment]::GetEnvironmentVariable($name)
  if (-not [string]::IsNullOrWhiteSpace($value)) { return $value }
  if ($dotenv.ContainsKey($name)) { return $dotenv[$name] }
  return $null
}

# Mirrors resolve_executable() / model_directory() in worker/separate_v4.py:
# env var, then anything vendored inside the project, then PATH. Keep in step.
function Resolve-Tool($envName, $bundled, $command) {
  $configured = Get-Setting $envName
  if (-not [string]::IsNullOrWhiteSpace($configured)) {
    return [pscustomobject]@{ Path = $configured; Source = "configured" }
  }
  foreach ($candidate in $bundled) {
    $resolved = Get-ChildItem -Path (Join-Path $project $candidate) -ErrorAction SilentlyContinue |
      Select-Object -First 1
    if ($resolved) {
      return [pscustomobject]@{ Path = $resolved.FullName; Source = "bundled" }
    }
  }
  $onPath = Get-Command $command -ErrorAction SilentlyContinue
  if ($onPath) { return [pscustomobject]@{ Path = $onPath.Source; Source = "PATH" } }
  return [pscustomobject]@{ Path = $null; Source = "not found" }
}

$separator = Resolve-Tool "STEMFLOW_SEPARATOR_EXE" @(
  ".sepenv\Scripts\audio-separator.exe",
  "vendor\sepenv\Scripts\audio-separator.exe"
) "audio-separator"

$ffmpeg = Resolve-Tool "STEMFLOW_FFMPEG_EXE" @(
  "vendor\ffmpeg\ffmpeg.exe",
  "vendor\ffmpeg\bin\ffmpeg.exe",
  "vendor\ffmpeg\*\bin\ffmpeg.exe"
) "ffmpeg"

$modelDir = Get-Setting "STEMFLOW_MODEL_DIR"
$modelSource = "configured"
if ([string]::IsNullOrWhiteSpace($modelDir)) {
  # Same candidates as model_directory(), all derived from the project root.
  $modelSource = "bundled"
  $modelDir = Join-Path $project "models"
  if (-not (Test-Path -LiteralPath (Join-Path $modelDir "download_checks.json"))) {
    $vendored = Join-Path $project "vendor\models"
    if (Test-Path -LiteralPath (Join-Path $vendored "download_checks.json")) {
      $modelDir = $vendored
    }
  }
}

Write-Output "Resolved components"
"  audio-separator  {0,-20} {1}" -f $separator.Source, $separator.Path
"  ffmpeg           {0,-20} {1}" -f $ffmpeg.Source, $ffmpeg.Path
"  model dir        {0,-20} {1}" -f $modelSource, $modelDir
Write-Output ""

# The .yaml configs and download_checks.json are as load-bearing as the
# checkpoints: without them audio-separator reaches out to the network even
# though the models are already on disk.
$checks = @(
  @{ Name = "audio-separator"; Path = $separator.Path },
  @{ Name = "FFmpeg"; Path = $ffmpeg.Path },
  @{ Name = "Vocal model"; Path = (Join-Path $modelDir "model_bs_roformer_ep_317_sdr_12.9755.ckpt") },
  @{ Name = "Vocal model config"; Path = (Join-Path $modelDir "model_bs_roformer_ep_317_sdr_12.9755.yaml") },
  @{ Name = "Six-stem model"; Path = (Join-Path $modelDir "BS-Roformer-SW.ckpt") },
  @{ Name = "Six-stem model config"; Path = (Join-Path $modelDir "BS-Roformer-SW.yaml") },
  @{ Name = "Model download index"; Path = (Join-Path $modelDir "download_checks.json") }
)

$failed = $false
foreach ($check in $checks) {
  $exists = ($null -ne $check.Path) -and (Test-Path -LiteralPath $check.Path)
  [pscustomobject]@{ Component = $check.Name; Ready = $exists; Path = $check.Path }
  if (-not $exists) { $failed = $true }
}

if ($failed) {
  throw "StemFlow GPU preflight failed. Every component above must exist locally for offline separation; see the Models section of README.md."
}

$python = Join-Path (Split-Path -Parent $separator.Path) "python.exe"
if (-not (Test-Path -LiteralPath $python)) {
  throw "Could not find python.exe next to audio-separator at $python."
}

& $python -c "import torch; assert torch.cuda.is_available(); print('CUDA:', torch.cuda.get_device_name(0))"

Write-Output "Offline separation assets verified in $modelDir."
