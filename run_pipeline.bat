@echo off
REM Run pipeline for AI-Powered Honeypot Intelligence System

echo ========================================
echo AI-Powered Honeypot Intelligence System
echo Pipeline Execution
echo ========================================
echo.

REM Check if virtual environment exists
if not exist "venv\Scripts\activate.bat" (
    echo ERROR: Virtual environment not found!
    echo Please run setup.bat first
    pause
    exit /b 1
)

REM Activate virtual environment
echo Activating virtual environment...
call venv\Scripts\activate.bat
echo.

REM Step 1: Data Preprocessing
echo ========================================
echo [1/3] Running Data Preprocessing...
echo ========================================
python src\data_preprocessing.py
if %errorlevel% neq 0 (
    echo ERROR: Data preprocessing failed!
    pause
    exit /b 1
)
echo.
echo Data preprocessing completed successfully!
echo Output saved to: output\processed_data.csv
echo.
pause

REM Step 2: Feature Engineering
echo ========================================
echo [2/3] Running Feature Engineering...
echo ========================================
python src\feature_engineering.py
if %errorlevel% neq 0 (
    echo ERROR: Feature engineering failed!
    pause
    exit /b 1
)
echo.
echo Feature engineering completed successfully!
echo Output saved to: output\feature_data.csv
echo.
pause

REM Step 3: Model Training
echo ========================================
echo [3/3] Running Model Training...
echo ========================================
python src\models.py
if %errorlevel% neq 0 (
    echo ERROR: Model training failed!
    pause
    exit /b 1
)
echo.
echo Model training completed successfully!
echo Models saved to: models\
echo.
pause

echo ========================================
echo Pipeline Completed Successfully!
echo ========================================
echo.
echo All processing steps completed.
echo.
echo To launch the dashboard, run:
echo    launch_dashboard.bat
echo.
echo Or manually run:
echo    venv\Scripts\activate
echo    streamlit run app\dashboard.py
echo.
pause
