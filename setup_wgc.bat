@echo off
setlocal
cd /d "%~dp0"
title Auto Maple WGC - First-time setup

where python >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python was not found in PATH.
    echo Install Python 3.13 and enable Add Python to PATH.
    pause
    exit /b 1
)

where dotnet >nul 2>nul
if errorlevel 1 (
    echo [ERROR] .NET 8 SDK was not found.
    pause
    exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
    echo [1/4] Creating Python virtual environment...
    python -m venv .venv || goto :failed
) else (
    echo [1/4] Existing Python virtual environment found.
)

echo [2/4] Updating pip...
".venv\Scripts\python.exe" -m pip install --upgrade pip || goto :failed

echo [3/4] Installing Python packages...
".venv\Scripts\python.exe" -m pip install -r requirements.txt || goto :failed

echo [4/4] Building MapleCaptureHost...
dotnet build capture_host\MapleCaptureHost.csproj -c Release || goto :failed

echo.
echo Setup completed successfully.
echo You can now double-click run_auto_maple_wgc.bat
pause
exit /b 0

:failed
echo.
echo [ERROR] Setup failed. Keep this window open and send the error text for review.
pause
exit /b 1
