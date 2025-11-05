# Quick Start Guide

## Step-by-Step Instructions

### Step 1: Setup (One-time only)

Navigate to the project directory and run the setup script:

**Windows:**
```cmd
setup.bat
```

**Linux/Mac:**
```bash
chmod +x setup.sh
./setup.sh
```

This will:
- Create a virtual environment (venv)
- Install all required Python packages
- Takes about 2-3 minutes

### Step 2: Run the Pipeline

After setup completes, run:

**Windows:**
```cmd
run_pipeline.bat
```

**Linux/Mac:**
```bash
./run_pipeline.sh
```

This will execute in order:
1. Data Preprocessing (3-5 seconds)
2. Feature Engineering (1-2 seconds)
3. Model Training (5-10 seconds)

The script will pause between steps so you can review the output.

### Step 3: Launch Dashboard

After the pipeline completes, run:

**Windows:**
```cmd
launch_dashboard.bat
```

**Linux/Mac:**
```bash
streamlit run app/dashboard.py
```

The dashboard will open automatically in your browser at http://localhost:8501

## Manual Commands (Alternative)

If you prefer to run commands manually:

**Windows:**
```cmd
# 1. Setup
python -m venv venv
venv\Scripts\activate
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn streamlit plotly

# 2. Run Pipeline
venv\Scripts\activate
python src\data_preprocessing.py
python src\feature_engineering.py
python src\models.py

# 3. Launch Dashboard
venv\Scripts\activate
streamlit run app\dashboard.py
```

**Linux/Mac:**
```bash
# 1. Setup
python3 -m venv venv
source venv/bin/activate
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn streamlit plotly

# 2. Run Pipeline
source venv/bin/activate
python src/data_preprocessing.py
python src/feature_engineering.py
python src/models.py

# 3. Launch Dashboard
source venv/bin/activate
streamlit run app/dashboard.py
```

## Expected Outputs

### After Data Preprocessing:
- Creates: `output/processed_data.csv`
- Shows: Number of records (70,327), unique IPs (3,508), unique ports (30,337)
- Protocol distribution across 10 types

### After Feature Engineering:
- Creates: `output/feature_data.csv`
- Shows: 19 features created, attack type distribution
- Severity level distribution (HIGH/MEDIUM/LOW)

### After Model Training:
- Creates: `models/attack_classifier_rf.pkl` (99.99% accuracy)
- Creates: `models/attack_classifier_xgb.pkl` (100% accuracy)
- Creates: `models/threat_scorer_rf.pkl` (99.99% accuracy)
- Creates: `models/threat_scorer_xgb.pkl` (100% accuracy)
- Shows: Model accuracy, precision, recall, F1-scores for all 4 models

### Dashboard:
- Opens browser window at http://localhost:8501
- Shows interactive analytics and visualizations
- 6 sections: Overview, Attack Analysis, Temporal Patterns, Threat Intelligence, ML Predictions, Advanced Analytics
