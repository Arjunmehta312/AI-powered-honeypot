@echo off
REM Setup script for AI-Powered Honeypot Intelligence System

echo ========================================
echo AI-Powered Honeypot Intelligence System
echo Setup Script
echo ========================================
echo.

REM Step 1: Create virtual environment
echo [1/4] Creating virtual environment...
python -m venv venv
if %errorlevel% neq 0 (
    echo ERROR: Failed to create virtual environment
    echo Please ensure Python 3.9+ is installed
    pause
    exit /b 1
)
echo Virtual environment created successfully!
echo.

REM Step 2: Activate virtual environment
echo [2/4] Activating virtual environment...
call venv\Scripts\activate.bat
echo Virtual environment activated!
echo.

REM Step 3: Upgrade pip
echo [3/4] Upgrading pip...
python -m pip install --upgrade pip
echo.

REM Step 4: Install requirements
echo [4/4] Installing dependencies...
pip install pandas numpy scikit-learn xgboost matplotlib seaborn plotly streamlit joblib
if %errorlevel% neq 0 (
    echo ERROR: Failed to install dependencies
    pause
    exit /b 1
)
echo.

echo ========================================
echo Setup Complete!
echo ========================================
echo.
echo Virtual environment created at: venv\
echo.
echo To activate the virtual environment, run:
echo    venv\Scripts\activate
echo.
echo To run the project, use: run_pipeline.bat
echo.
pause
