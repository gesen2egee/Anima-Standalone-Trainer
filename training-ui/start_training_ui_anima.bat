@echo off
setlocal
cd /d "%~dp0"

if not exist "package.json" (
    echo [ERROR] package.json not found.
    pause
    exit /b 1
)

if not exist "node_modules\" (
    echo [ERROR] Node dependencies not found. Please run setup.bat first.
    pause
    exit /b 1
)

where node >nul 2>nul
if errorlevel 1 (
    echo [ERROR] Node.js not found.
    pause
    exit /b 1
)

where npm >nul 2>nul
if errorlevel 1 (
    echo [ERROR] npm not found.
    pause
    exit /b 1
)

echo Starting Anima Training UI...
set "RUN_STATUS=0"
"..\venv\Scripts\python.exe" "launcher.py"
if errorlevel 1 set "RUN_STATUS=1"
echo.
echo Application exited (check for errors above).
pause
exit /b %RUN_STATUS%
