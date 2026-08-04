@echo off
setlocal
cd /d "%~dp0"

if not exist ".venv\Scripts\python.exe" (
    echo [!] 找不到 .venv，請先執行 setup_wgc.bat
    pause
    exit /b 1
)

echo.
echo ========================================
echo   Auto Maple Dataset Manager
echo ========================================
echo.
echo 將掃描 datasets\session_*，略過低品質資料並去除近似重複圖片。
echo 原始 Session 不會被刪除或修改。
echo.

".venv\Scripts\python.exe" tools\build_dataset.py
set "EXIT_CODE=%ERRORLEVEL%"

echo.
if "%EXIT_CODE%"=="0" (
    echo [OK] 完成。請查看 training_dataset 資料夾。
) else (
    echo [!] 建立失敗，錯誤碼：%EXIT_CODE%
)
echo.
pause
exit /b %EXIT_CODE%
