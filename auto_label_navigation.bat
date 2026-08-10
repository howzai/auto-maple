@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "tools\auto_label_navigation.py" goto :notool
if not exist "assets\models\navigation_scene.pt" goto :nomodel

"%PY%" -c "import ultralytics" >nul 2>&1
if errorlevel 1 goto :noyolo

echo.
echo ========================================
echo   Auto Maple Navigation Auto Label
echo ========================================
echo.
echo This will pre-label ladder/platform boxes on images that do not already
echo contain navigation labels. Existing monster labels are preserved.
echo.

"%PY%" "tools\auto_label_navigation.py" --split train --conf 0.45
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :failed

echo.
echo [OK] Navigation auto-label completed.
echo [OK] Open label_dataset.bat to review and correct the generated boxes.
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:notool
echo [ERROR] tools\auto_label_navigation.py was not found.
set "RC=1"
goto :done

:nomodel
echo [ERROR] assets\models\navigation_scene.pt was not found.
echo Run train_navigation_ai.bat first.
set "RC=1"
goto :done

:noyolo
echo [ERROR] YOLO packages are not installed.
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
echo.
echo [ERROR] Navigation auto-label stopped. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
