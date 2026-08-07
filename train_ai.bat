@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "tools\train_scene.py" goto :notool
if not exist "training_dataset\data.yaml" goto :nodataset

"%PY%" -c "import ultralytics" >nul 2>&1
if errorlevel 1 goto :noyolo

echo.
echo ========================================
echo   Auto Maple Scene AI Trainer
echo ========================================
echo.
echo This will train classic_scene.pt from training_dataset.
echo Training can take a long time depending on your GPU and dataset size.
echo.

"%PY%" "tools\train_scene.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :failed

echo.
echo [OK] Training completed.
echo [OK] Model: assets\models\classic_scene.pt
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:notool
echo [ERROR] tools\train_scene.py was not found.
set "RC=1"
goto :done

:nodataset
echo [ERROR] training_dataset\data.yaml was not found.
echo Run build_dataset.bat first.
set "RC=1"
goto :done

:noyolo
echo [ERROR] YOLO training packages are not installed.
echo Run setup_ai.bat first.
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
echo.
echo [ERROR] Training stopped. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
