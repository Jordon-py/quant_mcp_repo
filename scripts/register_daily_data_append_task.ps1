param(
    [string]$TaskName = "QuantMCP-DailyDataAppend",
    [string]$At = "05:15",
    [string]$Python = "C:\Python313\python.exe"
)

$ErrorActionPreference = "Stop"
$RepoRoot = Resolve-Path (Join-Path $PSScriptRoot "..")
$Runner = Join-Path $PSScriptRoot "run_daily_data_append.ps1"
$Argument = "-NoProfile -ExecutionPolicy Bypass -File `"$Runner`" -Python `"$Python`""

$Action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument $Argument -WorkingDirectory $RepoRoot
$Trigger = New-ScheduledTaskTrigger -Daily -At $At
$Settings = New-ScheduledTaskSettingsSet `
    -StartWhenAvailable `
    -MultipleInstances IgnoreNew `
    -RestartCount 3 `
    -RestartInterval (New-TimeSpan -Minutes 5) `
    -ExecutionTimeLimit (New-TimeSpan -Minutes 30)
$Principal = New-ScheduledTaskPrincipal -UserId $env:USERNAME -LogonType Interactive -RunLevel Limited

Register-ScheduledTask `
    -TaskName $TaskName `
    -Action $Action `
    -Trigger $Trigger `
    -Settings $Settings `
    -Principal $Principal `
    -Description "Append BTC/SOL Kraken OHLC datasets and update the local long-history archive for Quant Research MCP." `
    -Force | Out-Null

Get-ScheduledTask -TaskName $TaskName
Get-ScheduledTaskInfo -TaskName $TaskName
