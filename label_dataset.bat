@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] Python virtual environment not found.
    echo [~] Run setup_wgc.bat first.
    pause
    exit /b 1
)

if not exist "training_dataset\images\train" (
    echo [!] training_dataset was not found.
    echo [~] Run build_dataset.bat first.
    pause
    exit /b 1
)

".venv\Scripts\python.exe" tools\label_dataset.py --dataset training_dataset --split train
if errorlevel 1 (
    echo.
    echo [!] Label tool stopped with an error.
    pause
)
endlocal
