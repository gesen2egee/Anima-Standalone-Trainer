@echo off
cd /d %~dp0

node -v >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [INFO] Node.js is not installed. Attempting automatic installation...
    echo.

    where winget >nul 2>&1
    if %errorlevel% equ 0 (
        echo Installing Node.js via winget...
        winget install OpenJS.NodeJS --accept-source-agreements --accept-package-agreements
        if %errorlevel% neq 0 (
            echo.
            echo [ERROR] winget install failed.
            echo Please install Node.js manually from: https://nodejs.org/
            echo.
            pause
            exit /b 1
        )
        echo.
        echo [INFO] Node.js installed. Refreshing PATH...
        call RefreshEnv.cmd >nul 2>&1
        set "PATH=%PATH%;%ProgramFiles%\nodejs"
    ) else (
        echo winget not available. Downloading Node.js installer...
        echo.
        set "NODE_INSTALLER=%TEMP%\node_installer.msi"
        powershell -Command "$v = (Invoke-RestMethod 'https://nodejs.org/dist/index.json')[0].version; Invoke-WebRequest -Uri \"https://nodejs.org/dist/$v/node-$v-x64.msi\" -OutFile '%NODE_INSTALLER%'"
        if %errorlevel% neq 0 (
            echo.
            echo [ERROR] Failed to download Node.js installer.
            echo Please install manually from: https://nodejs.org/
            echo.
            pause
            exit /b 1
        )
        echo Running Node.js installer silently...
        msiexec /i "%NODE_INSTALLER%" /quiet /norestart
        if %errorlevel% neq 0 (
            echo.
            echo [ERROR] Node.js installation failed.
            echo Please install manually from: https://nodejs.org/
            echo.
            pause
            exit /b 1
        )
        del "%NODE_INSTALLER%" >nul 2>&1
        set "PATH=%PATH%;%ProgramFiles%\nodejs"
        echo Node.js installed successfully.
    )

    node -v >nul 2>&1
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Node.js still not found after installation.
        echo Please restart this script, or install Node.js manually and reopen a new terminal.
        echo Download from: https://nodejs.org/
        echo.
        pause
        exit /b 1
    )
)
echo Node.js detected.
echo.

python --version >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Python is not installed or not on PATH!
    echo Please install Python 3.10 - 3.13 from: https://www.python.org/downloads/
    echo Make sure to check "Add Python to PATH" during installation.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2 delims= " %%v in ('python --version 2^>^&1') do set PYVER=%%v
for /f "tokens=1,2 delims=." %%a in ("%PYVER%") do (
    set PY_MAJOR=%%a
    set PY_MINOR=%%b
)
if %PY_MAJOR% neq 3 (
    echo.
    echo [ERROR] Python 3.10 - 3.13 is required. Found Python %PYVER%.
    echo Please install a supported version from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
if %PY_MINOR% LSS 10 (
    echo.
    echo [ERROR] Python %PYVER% is too old. Minimum required: Python 3.10.
    echo Please install Python 3.10 - 3.13 from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
if %PY_MINOR% GEQ 14 (
    echo.
    echo [ERROR] Python %PYVER% is not yet supported. Maximum supported: Python 3.13.
    echo Please install Python 3.10 - 3.13 from: https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)
echo Python %PYVER% detected.
echo.

if not exist venv (
    echo Creating venv...
    python -m venv venv
    if %errorlevel% neq 0 (
        echo.
        echo [ERROR] Failed to create virtual environment.
        echo Make sure Python 3.10+ is installed correctly.
        echo.
        pause
        exit /b 1
    )
) else (
    echo Venv already exists.
)

set "VENV_PYTHON=venv\Scripts\python.exe"
if not exist "%VENV_PYTHON%" (
    echo.
    echo [ERROR] Virtual environment Python not found at %VENV_PYTHON%
    echo Try deleting the venv folder and running this script again.
    echo.
    pause
    exit /b 1
)

echo ----------------------------------------------------------------------
echo Upgrading pip and installing requirements...
echo ----------------------------------------------------------------------
"%VENV_PYTHON%" -m pip install --upgrade pip
if %errorlevel% neq 0 (
    echo [WARNING] Failed to upgrade pip. Continuing with requirements...
)

"%VENV_PYTHON%" -m pip install -r requirements.txt
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] pip install failed.
    echo Check the output above for details.
    echo.
    pause
    exit /b 1
)

echo.
echo ----------------------------------------------------------------------
echo Installing UI dependencies (npm install)...
echo ----------------------------------------------------------------------
cd training-ui
call npm install
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] npm install failed.
    echo Check the output above for details.
    echo.
    cd ..
    pause
    exit /b 1
)
cd ..

echo.
echo ----------------------------------------------------------------------
echo Installation Complete!
echo ----------------------------------------------------------------------
pause

