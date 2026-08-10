@echo off
setlocal EnableExtensions
set "ROOT=%~dp0"
pushd "%ROOT%" >nul 2>&1
if errorlevel 1 goto :badroot

set "PY=.venv\Scripts\python.exe"
if not exist "%PY%" goto :nopython
if not exist "tools\test_combined_vision.py" goto :notool
if not exist "assets\models\classic_scene.pt" goto :nomonster
if not exist "assets\models\navigation_scene.pt" goto :nonav

echo.
echo ========================================
echo   Auto Maple Combined Vision Test
echo ========================================
echo.
echo This test loads BOTH models:
echo   - classic_scene.pt      = monster
echo   - navigation_scene.pt   = ladder / platform
echo.
echo READ ONLY: no movement, attack, or key presses are sent to the game.
echo Open MapleStory first. Press F10 or Esc in the preview window to close.
echo.

"%PY%" "tools\test_combined_vision.py"
set "RC=%ERRORLEVEL%"
if not "%RC%"=="0" goto :failed
goto :done

:nopython
echo [ERROR] Python virtual environment was not found.
echo Run setup_wgc.bat first.
set "RC=1"
goto :done

:notool
echo [ERROR] tools\test_combined_vision.py was not found.
set "RC=1"
goto :done

:nomonster
echo [ERROR] Monster model not found: assets\models\classic_scene.pt
set "RC=1"
goto :done

:nonav
echo [ERROR] Navigation model not found: assets\models\navigation_scene.pt
set "RC=1"
goto :done

:badroot
echo [ERROR] Could not open the project directory.
set "RC=1"
goto :finish

:failed
echo.
echo [ERROR] Combined vision test stopped. Error code: %RC%

:done
popd >nul 2>&1
:finish
echo.
pause
exit /b %RC%
