@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Auto Maple - USB HID Shift Test

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment has not been created.
    echo Run setup_wgc.bat first.
    echo.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" -u "tools\test_hid_shift.py"
set "EXIT_CODE=%ERRORLEVEL%"
echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] HID Shift test completed.
) else (
    echo [ERROR] HID Shift test failed with exit code %EXIT_CODE%.
)
echo.
pause
exit /b %EXIT_CODE%
