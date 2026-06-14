@echo off
title Qythera - AI Superintelligence Installer

echo.
echo  ========================================
echo       Qythera - AI Superintelligence
echo         Universal Installer v1.0
echo  ========================================
echo.

REM Find Python
where python >nul 2>&1
if %ERRORLEVEL% neq 0 (
    echo ERROR: Python not found!
    echo Install from: https://www.python.org/downloads/
    pause
    exit /b 1
)

echo Python found: 
python --version

REM Check if project exists
if exist "core\__init__.py" (
    echo Project found in current directory.
) else if exist "qythera\core\__init__.py" (
    cd qythera
    echo Project found in .\qythera
) else (
    echo Downloading Qythera...
    git clone https://github.com/cpner/qythera.git
    cd qythera
)

REM Install numpy
echo Installing numpy...
pip install numpy --quiet 2>nul

echo.
echo ========================================
echo    Installation Complete!
echo ========================================
echo.
echo Quick start:
echo   python -m core.inference.server --port 8080
echo.
echo Or open web/standalone.html in browser
echo.
pause
