<#
.SYNOPSIS
  Register (or remove) a daily Windows Scheduled Task that runs scripts/daily_update.py.

.DESCRIPTION
  Creates a per-user task "WorldCup-DailyUpdate" that runs the project's venv Python on
  scripts/daily_update.py once a day, appending output to logs/daily_update.log.
  No administrator rights required (it runs as the current user).

.EXAMPLE
  powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1            # install at 09:00
  powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1 -At 21:30  # install at 21:30
  powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1 -Remove    # uninstall
  Get-ScheduledTask WorldCup-DailyUpdate ; Start-ScheduledTask WorldCup-DailyUpdate  # check / run now
#>
param(
    [string]$At = "09:00",
    [switch]$Remove
)

$ErrorActionPreference = "Stop"
$TaskName = "WorldCup-DailyUpdate"

$root   = Split-Path -Parent $PSScriptRoot
$py     = Join-Path $root ".venv\Scripts\python.exe"
$script = Join-Path $root "scripts\daily_update.py"
$logDir = Join-Path $root "logs"
$log    = Join-Path $logDir "daily_update.log"

if ($Remove) {
    if (Get-ScheduledTask -TaskName $TaskName -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $TaskName -Confirm:$false
        Write-Host "Removed scheduled task '$TaskName'."
    } else {
        Write-Host "No task named '$TaskName' to remove."
    }
    return
}

if (-not (Test-Path $py))     { throw "venv Python not found at $py - create it first (see README Setup)." }
if (-not (Test-Path $script)) { throw "daily_update.py not found at $script." }
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

# cmd /c wrapper so stdout+stderr are appended to the log. Extra outer quotes are
# required by cmd.exe when the command itself contains quoted paths.
$inner  = '""{0}" "{1}" >> "{2}" 2>&1"' -f $py, $script, $log
$action = New-ScheduledTaskAction -Execute "cmd.exe" -Argument "/c $inner" -WorkingDirectory $root
$trigger  = New-ScheduledTaskTrigger -Daily -At $At
$settings = New-ScheduledTaskSettingsSet -StartWhenAvailable -RunOnlyIfNetworkAvailable `
            -ExecutionTimeLimit (New-TimeSpan -Minutes 30)

Register-ScheduledTask -TaskName $TaskName -Action $action -Trigger $trigger -Settings $settings `
    -Description "Daily World Cup model: refresh API + eloratings data, regenerate predictions" -Force | Out-Null

Write-Host "Installed scheduled task '$TaskName' - runs daily at $At."
Write-Host "  Log:        $log"
Write-Host "  Run now:    Start-ScheduledTask $TaskName"
Write-Host "  Remove:     powershell -ExecutionPolicy Bypass -File scripts\setup_schedule.ps1 -Remove"
