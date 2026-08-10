@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "tools\train_navigation.py" goto :notool
if not exist "training_dataset\data.yaml" goto :nodataset

"%PY%" -c "import ultralytics, torch" >nul 2>&1
if errorlevel 1 goto :noyolo

echo.
echo ========================================
echo   Auto Maple Navigation AI Trainer
echo ========================================
echo.
echo Classes: ladder / platform
echo Player is intentionally excluded for now.
echo Monster model classic_scene.pt will NOT be modified.
echo Minimum first pass: 30 navigation-labeled images.
echo.

"%PY%" "tools\train_navigation.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :failed

echo.
echo [OK] Navigation training completed.
echo [OK] Model: assets\models\navigation_scene.pt
echo [OK] Next: run auto_label_navigation.bat
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:notool
echo [ERROR] tools\train_navigation.py was not found.
set "RC=1"
goto :done

:nodataset
echo [ERROR] training_dataset\data.yaml was not found.
echo Run build_dataset.bat first.
set "RC=1"
goto :done

:noyolo
echo [ERROR] YOLO/PyTorch packages are not installed correctly.
echo Run setup_ai.bat first.
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
echo.
echo [ERROR] Navigation training stopped. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
