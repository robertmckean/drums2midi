# codex_launch.ps1
# Purpose: Launch Codex from this workspace inside the drum310 Conda environment
# Features: Conda activation, workspace cwd setup, Python path echo
# Usage: powershell -ExecutionPolicy Bypass -File .\codex_launch.ps1

$ErrorActionPreference = "Stop"
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path

& conda 'shell.powershell' 'hook' | Out-String | Invoke-Expression
conda activate drum310

if ($env:CONDA_DEFAULT_ENV -ne "drum310") {
    throw "Failed to activate Conda environment 'drum310'."
}

$pythonExe = (Get-Command python -ErrorAction Stop).Source
Write-Host "Activated Conda environment: $env:CONDA_DEFAULT_ENV"
Write-Host "Python: $pythonExe"

Set-Location $scriptDir
codex
