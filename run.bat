@echo off
cd /d "%~dp0"
if not exist ".venv\Scripts\python.exe" (
    echo [ERROR] setup.bat must be run first.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" factory_app.py
if errorlevel 1 pause
