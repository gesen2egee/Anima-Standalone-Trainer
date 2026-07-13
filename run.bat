@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"

echo.
echo [Run] 啟動 Anima Training UI
echo 專案路徑: "%ROOT%"
echo.

if not exist "%ROOT%training-ui\package.json" (
    echo [ERROR] 找不到 training-ui\package.json，請確認此檔案放在專案根目錄。
    pause
    exit /b 1
)

if not exist "%ROOT%training-ui\node_modules\" (
    echo [ERROR] 尚未安裝 Training UI Node dependencies。
    echo 請先執行 setup.bat，完成後再執行 run.bat。
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 找不到 npm。請先安裝 Node.js LTS，並重新開啟終端機。
    pause
    exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 找不到 node。請先安裝 Node.js LTS，並重新開啟終端機。
    pause
    exit /b 1
)

call npm --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm 無法執行，請重新安裝 Node.js LTS。
    pause
    exit /b 1
)

if not exist "%ROOT%venv\Scripts\python.exe" (
    echo [ERROR] 找不到 Python venv。
    echo 請先執行 setup.bat，完成 Python 環境安裝後再執行 run.bat。
    pause
    exit /b 1
)

"%ROOT%venv\Scripts\python.exe" -c "import torch, accelerate" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Python venv 缺少必要套件。
    echo 請重新執行 setup.bat 完成安裝。
    pause
    exit /b 1
)

pushd "%ROOT%training-ui"
if errorlevel 1 (
    echo [ERROR] 無法進入 training-ui 資料夾。
    pause
    exit /b 1
)

echo 正在啟動 UI...
set "RUN_STATUS=0"
call npm start
if errorlevel 1 set "RUN_STATUS=1"
popd

echo.
echo UI 已結束，Exit Code: %RUN_STATUS%
pause
exit /b %RUN_STATUS%
