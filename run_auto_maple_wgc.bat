@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title Auto Maple - Windows Graphics Capture

echo.
echo ========================================
echo   Auto Maple - Windows Graphics Capture
echo ========================================
echo Project: %CD%
echo.

if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] Python environment has not been created.
    echo Run setup_wgc.bat first.
    echo.
    pause
    exit /b 1
)

if not exist "main.py" (
    echo [ERROR] main.py was not found in:
    echo %CD%
    echo.
    pause
    exit /b 1
)

if not exist "capture_host\bin\Release\net8.0-windows10.0.19041.0\win-x64\MapleCaptureHost.exe" (
    echo [~] MapleCaptureHost is not built. Building now...
    dotnet build capture_host\MapleCaptureHost.csproj -c Release
    if errorlevel 1 (
        echo.
        echo [ERROR] MapleCaptureHost build failed.
        pause
        exit /b 1
    )
)

echo [~] Python executable:
echo     %CD%\.venv\Scripts\python.exe
echo [~] Importing local main module explicitly...
echo.

".venv\Scripts\python.exe" -u -X faulthandler -c "import os,sys,main; print('[DIAG] cwd:', os.getcwd()); print('[DIAG] python:', sys.executable); print('[DIAG] main module:', main.__file__); print('[DIAG] has main():', hasattr(main, 'main')); print('[DIAG] calling main.main() now...'); main.main(); print('[DIAG] main.main() returned normally')"
set "EXIT_CODE=%ERRORLEVEL%"

echo.
echo ========================================
echo Auto Maple Python process has ended.
echo Exit code: %EXIT_CODE%
echo ========================================

if not "%EXIT_CODE%"=="0" (
    echo [ERROR] Auto Maple exited abnormally.
    echo The traceback above is the primary diagnostic.
    echo Also check logs\auto-maple.log if it exists.
) else (
    echo [INFO] Python returned exit code 0.
    echo If you saw '[DIAG] main.main() returned normally' without a GUI,
    echo then the application startup path itself returned and must be fixed next.
)

echo.
echo Press any key to close this window.
pause >nul
exit /b %EXIT_CODE%
