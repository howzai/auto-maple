@echo off
setlocal
cd /d "%~dp0"

echo ========================================
echo   Auto Maple Monster Auto Label
echo ========================================
echo.

if exist ".venv\Scripts\python.exe" (
    set "PY=.venv\Scripts\python.exe"
) else (
    set "PY=python"
)

if not exist "assets\models\classic_scene.pt" (
    echo [ERROR] assets\models\classic_scene.pt was not found.
    echo Train the first monster model with train_ai.bat first.
    echo.
    pause
    exit /b 1
)

"%PY%" tools\auto_label.py --dataset training_dataset --split train --conf 0.72
set "ERR=%ERRORLEVEL%"

echo.
if not "%ERR%"=="0" (
    echo [ERROR] Auto label failed with code %ERR%.
) else (
    echo [OK] Auto label completed.
    echo Open label_dataset.bat to review and correct the generated boxes.
)
echo.
pause
exit /b %ERR%
