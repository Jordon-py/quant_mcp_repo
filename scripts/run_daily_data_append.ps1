param(
    [string]$Python = "C:\Python313\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
Set-Location $RepoRoot

$env:PYTHONPATH = Join-Path $RepoRoot "src"
$env:PYTHONDONTWRITEBYTECODE = "1"

& $Python -m quant_mcp.ops.daily_data_append --symbols BTC/USD SOL/USD --interval-minutes 60
if ($LASTEXITCODE -ne 0) {
    exit $LASTEXITCODE
}

& $Python -m quant_mcp.ops.history_archive --symbols BTC/USD SOL/USD --interval-minutes 60
exit $LASTEXITCODE
