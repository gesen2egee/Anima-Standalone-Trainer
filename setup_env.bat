@echo off
cd /d %~dp0

node -v >nul 2>&1
if %errorlevel% neq 0 (
    echo.
    echo [ERROR] Node.js is not installed!
    echo Node.js is required for the Training UI.
    echo Please download and install it from: https://nodejs.org/
    echo.
    pause
    exit /b
)
echo Node.js detected.
echo.

if not exist venv (
    echo Creating venv...
    python -m venv venv
) else (
    echo Venv already exists.
)

call venv\Scripts\activate.bat

echo ----------------------------------------------------------------------
echo Installing requirements from requirements.txt...
echo ----------------------------------------------------------------------
pip install -r requirements.txt

echo.


echo.
echo ----------------------------------------------------------------------
echo Installing UI dependencies (npm install)...
echo ----------------------------------------------------------------------
cd training-ui
call npm install
cd ..

echo.
echo ----------------------------------------------------------------------
echo Installation Complete!
echo ----------------------------------------------------------------------
pause

