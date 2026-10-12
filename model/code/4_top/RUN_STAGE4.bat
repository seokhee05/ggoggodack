@echo off
setlocal
cd /d "%~dp0"

if "%~1"=="" (
  echo.
  echo Usage:
  echo   RUN_STAGE4.bat "C:\path\input_video.mp4"
  echo.
  pause
  exit /b 1
)

if not exist ".venv\Scripts\python.exe" (
  echo [1/3] Creating virtual environment...
  py -3.11 -m venv .venv 2>nul || py -3 -m venv .venv
  if errorlevel 1 (
    echo Failed to create virtual environment. Install Python 3.10 or 3.11 first.
    pause
    exit /b 1
  )
  echo [2/3] Installing packages...
  .venv\Scripts\python.exe -m pip install --upgrade pip
  .venv\Scripts\python.exe -m pip install -r requirements.txt
)

echo [3/3] Running Stage-4...
.venv\Scripts\python.exe step4_glove_pose_dashboard_v15_final.py ^
  --video "%~1" ^
  --seg-model "armrest_3class_best.pt" ^
  --glove-model "glove_runs\glove_detector_v2\weights\best.pt" ^
  --full ^
  --out "results"

echo.
echo Finished. Check the results folder.
pause
