@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "VENV_DIR=%ROOT%venv"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"
set "REQ_FILE=%ROOT%requirements.txt"

echo.
echo [Setup] Anima Standalone Trainer 一鍵安裝
echo 專案路徑: "%ROOT%"
echo.

if not exist "%REQ_FILE%" (
    echo [ERROR] 找不到 requirements.txt，請確認此檔案放在專案根目錄。
    goto :fail
)

if not exist "%VENV_PY%" (
    echo [1/4] 建立 Python venv...
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3 -m venv "%VENV_DIR%"
    ) else (
        where python >nul 2>nul
        if errorlevel 1 (
            echo [ERROR] 找不到 Python。請先安裝 Python 3.10+，並勾選 Add Python to PATH。
            goto :fail
        )
        python -m venv "%VENV_DIR%"
    )
    if errorlevel 1 goto :fail
) else (
    echo [1/4] 已找到現有 venv，略過建立。
)

echo [2/4] 更新 pip / setuptools / wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [3/4] 安裝 Python requirements...
"%VENV_PY%" -m pip install -r "%REQ_FILE%"
if errorlevel 1 goto :fail

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 找不到 npm。請先安裝 Node.js LTS，並重新開啟終端機。
    goto :fail
)

echo [4/4] 安裝 Training UI Node dependencies...
pushd "%ROOT%training-ui"
if errorlevel 1 goto :fail
npm install
if errorlevel 1 (
    popd
    goto :fail
)
popd

echo.
echo [OK] 安裝完成。之後可執行 run.bat 開啟 UI。
pause
exit /b 0

:fail
echo.
echo [FAILED] 安裝未完成，請查看上方錯誤訊息。
pause
exit /b 1
