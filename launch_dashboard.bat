@echo off
REM Launch Dashboard for AI-Powered Honeypot Intelligence System

echo ========================================
echo AI-Powered Honeypot Intelligence System
echo Dashboard Launcher
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Check if processed data exists
if not exist "output\processed_data.csv" (
    echo WARNING: Processed data not found!
    echo Please run run_pipeline.bat first
    pause
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Launch dashboard
echo Launching Streamlit Dashboard...
echo.
echo The dashboard will open in your default browser at:
echo http://localhost:8501
echo.
echo Press Ctrl+C to stop the dashboard
echo.
streamlit run app\dashboard.py
