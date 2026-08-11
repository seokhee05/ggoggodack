@echo off
setlocal
cd /d "%~dp0"

echo ==================================================
echo  Armrest Realtime Inspection - Environment Setup
echo ==================================================
echo.

where winget >nul 2>&1
if errorlevel 1 (
    echo [ERROR] winget is not available.
    echo Install "App Installer" from Microsoft Store and run this file again.
    pause
    exit /b 1
)

echo [1/5] Installing Python 3.11...
winget install --id Python.Python.3.11 --exact --silent ^
    --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :install_error

echo [2/5] Installing Git...
winget install --id Git.Git --exact --silent ^
    --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :install_error

echo [3/5] Installing Visual Studio Code...
winget install --id Microsoft.VisualStudioCode --exact --silent ^
    --accept-package-agreements --accept-source-agreements
if errorlevel 1 goto :install_error

set "PYTHON_EXE=%LocalAppData%\Programs\Python\Python311\python.exe"

if not exist "%PYTHON_EXE%" (
    for /f "delims=" %%P in ('where py 2^>nul') do (
        set "PY_LAUNCHER=%%P"
        goto :python_found
    )
    echo [ERROR] Python was installed, but python.exe could not be located.
    echo Close this CMD window, open a new one, and run setup.bat again.
    pause
    exit /b 1
)

:python_found
echo [4/5] Creating Python virtual environment...
if exist "%PYTHON_EXE%" (
    "%PYTHON_EXE%" -m venv .venv
) else (
    "%PY_LAUNCHER%" -3.11 -m venv .venv
)
if errorlevel 1 goto :install_error

echo [5/5] Installing Python packages...
".venv\Scripts\python.exe" -m pip install --upgrade pip
if errorlevel 1 goto :install_error

".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 goto :install_error

echo.
echo ==================================================
echo  Setup completed successfully.
echo ==================================================
echo.
echo Run the program with:
echo   .venv\Scripts\python.exe factory_app.py
echo.
echo Open the project in VS Code with:
echo   code .
echo.
pause
exit /b 0

:install_error
echo.
echo [ERROR] Installation failed.
echo Check the messages above and your internet connection.
pause
exit /b 1
