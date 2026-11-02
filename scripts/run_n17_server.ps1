# Windows PowerShell 5.1 / PowerShell 7. ASCII wrapper for encoding portability.
param(
    [ValidateSet('check','pilot','formal','collect')][string]$Action = 'check',
    [string]$Config = '',
    [string]$Run = '',
    [switch]$DryRun,
    [switch]$Setup
)
$ErrorActionPreference = 'Stop'
$projectRoot = Split-Path -Parent $PSScriptRoot
$env:PYTHONUTF8 = '1'
$env:PYTHONIOENCODING = 'utf-8'
if ($Setup) {
    if (!(Get-Command uv -ErrorAction SilentlyContinue)) {
        throw 'Install uv first (for example: py -m pip install uv), then rerun -Setup'
    }
    Push-Location -LiteralPath $projectRoot
    try {
        & uv sync --locked --python 3.11 --group training --group perception --group simulation-video
        if ($LASTEXITCODE -ne 0) { exit $LASTEXITCODE }
    } finally { Pop-Location }
    Write-Output 'Environment prepared. Fill configs/experiments/n17_server.yaml and run -Action check.'
    exit 0
}
$pythonPath = Join-Path $projectRoot '.venv\Scripts\python.exe'
if (!(Test-Path -LiteralPath $pythonPath)) { throw 'Missing .venv. Run this script with -Setup first.' }
$arguments = @((Join-Path $PSScriptRoot 'run_n17_server.py'), '--action', $Action)
if ($Config) { $arguments += @('--config', $Config) }
if ($Run) { $arguments += @('--run', $Run) }
if ($DryRun) { $arguments += '--dry-run' }
& $pythonPath @arguments
exit $LASTEXITCODE
