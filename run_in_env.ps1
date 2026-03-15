# run_in_env.ps1
# Purpose: Run a project Python entry point inside the drum310 Conda environment
# Features: Conda activation, workspace cwd setup, source-path preflight checks
# Usage: powershell -ExecutionPolicy Bypass -File .\run_in_env.ps1 src\infer.py [args...]

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

if ($args.Count -lt 1) {
    throw "Usage: run_in_env.ps1 <script_path> [args...]"
}

$scriptPath = $args[0]
$scriptArgs = @()
if ($args.Count -gt 1) {
    $scriptArgs = $args[1..($args.Count - 1)]
}

& conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
conda activate drum310

if ($env:CONDA_DEFAULT_ENV -ne "drum310") {
    throw "Failed to activate Conda environment 'drum310'."
}

Set-Location $scriptDir

$configPath = Join-Path $scriptDir "config.py"
if (-not (Test-Path $configPath)) {
    throw "Missing config.py in workspace root."
}

$sourceDir = "C:\Users\windo\OneDrive\Python\drums_to_midi\00_GM"
if (-not (Test-Path $sourceDir)) {
    throw "Missing source data folder: $sourceDir"
}

$pythonExe = (Get-Command python -ErrorAction Stop).Source
Write-Host "Activated Conda environment: $env:CONDA_DEFAULT_ENV"
Write-Host "Python: $pythonExe"
Write-Host "Working directory: $scriptDir"
Write-Host "Running: python $scriptPath $scriptArgs"

& python $scriptPath @scriptArgs
