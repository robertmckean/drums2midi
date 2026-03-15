@echo off
setlocal
set "SCRIPT_DIR=%~dp0"
if "%SCRIPT_DIR:~-1%"=="\" set "SCRIPT_DIR=%SCRIPT_DIR:~0,-1%"

for /f %%i in ('powershell -NoProfile -Command "Get-Date -Format yyyyMMdd_HHmmss"') do set timestamp=%%i

set "log_dir=%SCRIPT_DIR%\codex_logs"
set "log_file=%log_dir%\codex_session_%timestamp%.txt"

if not exist "%log_dir%" mkdir "%log_dir%"

echo Codex Code Session - %date% %time% > "%log_file%"
echo ================================================ >> "%log_file%"
echo Workspace: %SCRIPT_DIR% >> "%log_file%"
echo. >> "%log_file%"
echo Notes: >> "%log_file%"
echo. >> "%log_file%"
echo ================================================ >> "%log_file%"
echo End of session >> "%log_file%"
echo ================================================ >> "%log_file%"
echo Created placeholder log file: %log_file%
echo.

start notepad "%log_file%"

echo Launching Codex Code...
powershell -NoProfile -ExecutionPolicy Bypass -Command "Start-Process powershell.exe -WorkingDirectory '%SCRIPT_DIR%' -ArgumentList '-NoExit','-ExecutionPolicy','Bypass','-File','%SCRIPT_DIR%\codex_launch.ps1'"
echo.
echo Codex Code launched in new window. Notepad is open for session notes.
endlocal
