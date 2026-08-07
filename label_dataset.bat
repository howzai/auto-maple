@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "tools\label_dataset.py" goto :notool
if not exist "training_dataset\images\train" goto :nodataset

echo.
echo ========================================
echo   Auto Maple Dataset Labeler
echo ========================================
echo.
echo Controls:
echo   1 player  2 monster  3 ladder  4 platform  5 obstacle
echo   Drag left mouse: add box
echo   Right click: select box
echo   Delete: remove selected box
echo   Space: save and next image
echo.

"%PY%" "tools\label_dataset.py" --dataset "training_dataset" --split train
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :failed
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:notool
echo [ERROR] tools\label_dataset.py was not found.
set "RC=1"
goto :done

:nodataset
echo [ERROR] training_dataset\images\train was not found.
echo Run build_dataset.bat first.
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
echo.
echo [ERROR] Label tool stopped. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
