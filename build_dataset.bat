@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "tools\build_dataset.py" goto :notool

echo.
echo ========================================
echo   Auto Maple Dataset Manager
echo ========================================
echo.
echo Scanning datasets\session_* ...
echo Source sessions will NOT be modified.
echo.

"%PY%" "tools\build_dataset.py"
set "RC=%ERRORLEVEL%"
echo.
if not "%RC%"=="0" goto :failed
echo [OK] Dataset build completed.
echo [OK] Output: training_dataset
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:notool
echo [ERROR] tools\build_dataset.py was not found.
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
echo [ERROR] Dataset build failed. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
