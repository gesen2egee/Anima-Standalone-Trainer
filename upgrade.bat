@echo off
chcp 65001 >nul
setlocal

set "ROOT=%~dp0"
set "VENV_DIR=%ROOT%venv"
set "VENV_PY=%ROOT%venv\Scripts\python.exe"
set "REQ_FILE=%ROOT%requirements.txt"
set "PPOCR_REQ_FILE=%ROOT%requirements-ppocr.txt"
set "TORCH_REQ_FILE=%ROOT%requirements-cu128.txt"
set "ACCELERATE_CONFIG=%USERPROFILE%\.cache\huggingface\accelerate\default_config.yaml"
if defined HF_HOME set "ACCELERATE_CONFIG=%HF_HOME%\accelerate\default_config.yaml"
set "ROOT_PUSHED=0"
set "VENV_CREATED=0"

pushd "%ROOT%"
if errorlevel 1 goto :fail
set "ROOT_PUSHED=1"

echo.
echo [Upgrade] Anima Standalone Trainer 一鍵升級
echo 專案路徑: "%ROOT%"
echo.

if exist "%ROOT%.git\" (
    where git >nul 2>nul
    if errorlevel 1 (
        echo [ERROR] 找不到 git，無法更新程式碼。
        goto :fail
    )
    echo [1/8] 更新 Git 程式碼...
    git pull --ff-only
    if errorlevel 1 goto :fail
) else (
    echo [1/8] 不是 Git 工作目錄，略過程式碼更新。
)

if not exist "%REQ_FILE%" (
    echo [ERROR] 找不到 requirements.txt，請確認此檔案放在專案根目錄。
    goto :fail
)

if not exist "%TORCH_REQ_FILE%" (
    echo [ERROR] 找不到 requirements-cu128.txt，請確認 CUDA 12.8 依賴檔放在專案根目錄。
    goto :fail
)

if not exist "%PPOCR_REQ_FILE%" (
    echo [ERROR] 找不到 requirements-ppocr.txt，請確認 PPOCR 依賴檔放在專案根目錄。
    goto :fail
)

if not exist "%VENV_PY%" (
    echo [2/8] 建立 Python venv...
    set "VENV_CREATED=1"
    set "PYTHON_CMD="
    where py >nul 2>nul
    if not errorlevel 1 (
        py -3.10 -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
        if not errorlevel 1 set "PYTHON_CMD=py -3.10"
        if not defined PYTHON_CMD (
            py -3 -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
            if not errorlevel 1 set "PYTHON_CMD=py -3"
        )
    )
    if not defined PYTHON_CMD (
        where python >nul 2>nul
        if not errorlevel 1 (
            python -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
            if not errorlevel 1 set "PYTHON_CMD=python"
        )
    )
    if not defined PYTHON_CMD (
        echo [ERROR] 找不到可用的 Python 3.10-3.12。README 以 Python 3.10.x 為基準，請先安裝 Python。
        goto :fail
    )
    %PYTHON_CMD% -m venv "%VENV_DIR%"
    if errorlevel 1 goto :fail
) else (
    echo [2/8] 已找到現有 venv，略過建立。
)

"%VENV_PY%" -c "import sys; raise SystemExit(0 if (3, 10) <= sys.version_info[:2] < (3, 13) else 1)" >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 現有 venv 無法使用，或 Python 版本不在 3.10-3.12。請刪除 venv 後重新執行 upgrade.bat。
    goto :fail
)

echo [3/8] 更新 pip / setuptools / wheel...
"%VENV_PY%" -m pip install --upgrade pip setuptools wheel
if errorlevel 1 goto :fail

echo [4/8] 更新 CUDA 12.8 PyTorch requirements...
"%VENV_PY%" -m pip install --upgrade -r "%TORCH_REQ_FILE%"
if errorlevel 1 goto :fail

echo [5/8] 更新 Python requirements...
"%VENV_PY%" -m pip install --use-pep517 --upgrade -r "%REQ_FILE%"
if errorlevel 1 goto :fail

echo [6/8] 更新 PP-OCRv6 requirements 並保留 OpenCV 5.0...
"%VENV_PY%" -m pip install --upgrade --no-deps -r "%PPOCR_REQ_FILE%"
if errorlevel 1 goto :fail
"%VENV_PY%" -c "import cv2, paddleocr, paddlex; raise SystemExit(0 if cv2.__version__ == '5.0.0' else 1)"
if errorlevel 1 (
    echo [ERROR] PP-OCRv6 安裝驗證失敗，或實際載入的 OpenCV 不是 5.0.0。
    goto :fail
)

echo [7/8] 確認 Accelerate 設定...
if "%VENV_CREATED%"=="1" (
    if exist "%ACCELERATE_CONFIG%" (
        echo 已找到 Accelerate 設定，略過互動設定。
    ) else (
        echo venv 是此次升級時新建立，請依 README 完成 Accelerate 設定：
        echo This machine / No distributed training / NO / NO / NO / all / fp16
        "%VENV_DIR%\Scripts\accelerate.exe" config
        if errorlevel 1 goto :fail
    )
) else (
    echo 已存在 venv，依 README 升級流程保留現有 Accelerate 設定。
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 找不到 npm。請先安裝 Node.js LTS，並重新開啟終端機。
    goto :fail
)

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] 找不到 node。請先安裝 Node.js LTS，並重新開啟終端機。
    goto :fail
)

call npm --version >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm 無法執行，請重新安裝 Node.js LTS。
    goto :fail
)

if not exist "%ROOT%training-ui\package.json" (
    echo [ERROR] 找不到 training-ui\package.json。
    goto :fail
)

echo [8/8] 更新 Training UI Node dependencies...
pushd "%ROOT%training-ui"
if errorlevel 1 goto :fail
call npm install
if errorlevel 1 (
    popd
    goto :fail
)
popd

echo.
echo [OK] 升級完成。可執行 run.bat 開啟 UI。
if "%ROOT_PUSHED%"=="1" popd
pause
exit /b 0

:fail
if "%ROOT_PUSHED%"=="1" popd
echo.
echo [FAILED] 升級未完成，請查看上方錯誤訊息。
pause
exit /b 1
