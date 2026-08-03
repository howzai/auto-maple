@echo off
setlocal
cd /d "%~dp0"
title Auto Maple - Windows Graphics Capture

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment has not been created.
    echo Double-click setup_wgc.bat first.
    pause
    exit /b 1
)

if not exist "capture_host\bin\Release\net8.0-windows10.0.19041.0\win-x64\MapleCaptureHost.exe" (
    echo MapleCaptureHost is not built. Building now...
    dotnet build capture_host\MapleCaptureHost.csproj -c Release
    if errorlevel 1 (
        echo [ERROR] MapleCaptureHost build failed.
        pause
        exit /b 1
    )
)

".venv\Scripts\python.exe" main.py
set "EXIT_CODE=%ERRORLEVEL%"

if not "%EXIT_CODE%"=="0" (
    echo.
    echo Auto Maple exited with code %EXIT_CODE%.
    echo Review logs\auto-maple.log for details.
    pause
)

exit /b %EXIT_CODE%
