@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "requirements-ai.txt" goto :noreq

echo.
echo ========================================
echo   Auto Maple AI Training Setup
echo ========================================
echo.
echo Installing optional YOLO training packages...
"%PY%" -m pip install --upgrade pip
if errorlevel 1 goto :failed
"%PY%" -m pip install -r "requirements-ai.txt"
if errorlevel 1 goto :failed

echo.
echo [OK] AI training environment is ready.
set "RC=0"
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:noreq
echo [ERROR] requirements-ai.txt was not found.
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
set "RC=%ERRORLEVEL%"
echo.
echo [ERROR] AI training setup failed. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
