# User Guide - AI-Powered Honeypot Intelligence System

## Table of Contents

1. [Installation](#installation)
2. [Quick Start](#quick-start)
3. [Data Processing](#data-processing)
4. [Model Training](#model-training)
5. [Dashboard Usage](#dashboard-usage)
6. [Troubleshooting](#troubleshooting)

## Installation

### Prerequisites

- Python 3.9 or higher installed
- At least 4GB RAM available
- 2GB free disk space
- Internet connection for package installation

### Step 1: Create Virtual Environment and Install Dependencies

Navigate to the project directory in your terminal:

**Windows:**
```powershell
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Install required packages
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn streamlit plotly
```

**Linux/Mac:**
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Install required packages
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn streamlit plotly
```

**Note**: Installation may take 2-3 minutes. PyArrow is optional and can be skipped if build errors occur.

This will install all necessary dependencies including:
- pandas, numpy (data processing)
- scikit-learn 1.7.2 (Random Forest)
- xgboost 1.7.6 (Gradient Boosting)
- joblib 1.5.2 (Model persistence)
- streamlit 1.51.0 (Web dashboard)
- plotly 5.17.0 (Interactive visualizations)
- matplotlib, seaborn (Static visualizations)

### Step 2: Verify Installation

Verify Python and packages are correctly installed:

```powershell
python --version
python -c "import pandas, sklearn, xgboost, streamlit; print('All packages installed successfully')"
```

## Quick Start

### Running the Complete Pipeline

#### Option 1: Automated Script (Recommended)

**Windows:**
```powershell
# Activate virtual environment first
venv\Scripts\activate

# Run complete pipeline
run_pipeline.bat
```

**Linux/Mac:**
```bash
# Activate virtual environment first
source venv/bin/activate

# Run complete pipeline (you may need to create this script or run steps individually)
python src/data_preprocessing.py
python src/feature_engineering.py
python src/models.py
```

This will execute all steps sequentially with progress updates.

#### Option 2: Individual Steps

Execute all steps in sequence:

**Windows:**
```powershell
# Activate virtual environment
venv\Scripts\activate

# Step 1: Data Preprocessing (3-5 seconds)
python src\data_preprocessing.py

# Step 2: Feature Engineering (1-2 seconds)
python src\feature_engineering.py

# Step 3: Model Training (5-10 seconds)
python src\models.py

# Step 4: Launch Dashboard
streamlit run app\dashboard.py
```

**Linux/Mac:**
```bash
# Activate virtual environment
source venv/bin/activate

# Step 1: Data Preprocessing (3-5 seconds)
python src/data_preprocessing.py

# Step 2: Feature Engineering (1-2 seconds)
python src/feature_engineering.py

# Step 3: Model Training (5-10 seconds)
python src/models.py

# Step 4: Launch Dashboard
streamlit run app/dashboard.py
```

The dashboard will automatically open in your default web browser at `http://localhost:8501`

## Data Processing

### Running Data Preprocessing

The preprocessing module handles data cleaning and initial feature extraction.

```powershell
python src\data_preprocessing.py
```

#### What This Does:

1. Loads raw data from `dataset/london.csv`
2. Removes duplicate records
3. Cleans IP addresses and port numbers
4. Parses timestamps into datetime objects
5. Extracts temporal features (hour, day, day of week)
6. Categorizes payload types (BINARY, TEXT, EMPTY)
7. Identifies protocols from payload signatures
8. Calculates frequency statistics
9. Saves processed data to `output/processed_data.csv`

#### Expected Output:

```
INFO - Loading data from dataset\london.csv
INFO - Loaded 70338 records
INFO - Cleaning data...
INFO - Removed 9 duplicate records
INFO - Found 5 missing values
INFO - Dropped rows with missing critical fields
INFO - Cleaning complete. 70327 records remaining
INFO - Parsing timestamps...
INFO - Extracting payload features...
INFO - Calculating frequency features...
INFO - Identifying protocols...
INFO - Protocol identification complete
INFO - Processed data saved to output\processed_data.csv

Preprocessing Complete!
Total Records: 70327
Unique IPs: 3508
Unique Ports: 30337

Protocol Distribution:
UNKNOWN: 21550
HTTP: 16272
TLS_SSL: 11344
RDP: 10721
EMPTY: 5100
SMB: 3368
SSH: 947
REDIS: 535
ADB: 269
SIP: 221
```

#### Output Files:

- `output/processed_data.csv` - Cleaned and enriched data
- `output/honeypot_intelligence.log` - Processing logs

### Running Feature Engineering

The feature engineering module creates advanced features for machine learning.

```powershell
python src\feature_engineering.py
```

#### What This Does:

1. Loads processed data
2. Creates attack type labels (8 categories)
3. Assigns threat severity levels (LOW, MEDIUM, HIGH)
4. Calculates IP reputation scores (0-100)
5. Encodes categorical variables
6. Creates interaction features
7. Builds time-based features
8. Saves engineered features to `output/feature_data.csv`

#### Expected Output:

```
INFO - Creating attack labels...
INFO - Attack type distribution:
UNKNOWN            20568
HTTP_EXPLOIT       16272
TLS_SSL            11344
RDP_ATTACK         10721
PORT_SCAN           6571
SMB_ATTACK          3368
SSH_BRUTE_FORCE      947
DATABASE_ATTACK      536

INFO - Creating severity labels...
INFO - Severity distribution:
HIGH      60116
MEDIUM     8177
LOW        2034

Feature Engineering Complete!
Total Features: 19
Engineered data saved to: output\feature_data.csv
```

#### Output Files:

- `output/feature_data.csv` - Complete feature set with 19 features
- `models/label_encoder.pkl` - Saved label encoders

## Model Training

### Training Machine Learning Models

```powershell
python src\models.py
```

#### What This Does:

1. Loads engineered features
2. Splits data into training and test sets (80/20)
3. Trains Random Forest attack classifier
4. Trains XGBoost attack classifier
5. Trains Random Forest threat severity scorer
6. Trains XGBoost threat severity scorer
7. Evaluates all models with cross-validation
8. Saves trained models and evaluation metrics

#### Expected Output:

```
================================================================================
MODEL EVALUATION SUMMARY
================================================================================
                Model  Accuracy  Precision   Recall  F1 Score CV Mean CV Std
 attack_classifier_rf  0.9999    0.9999     0.9999   0.9999   None    None
attack_classifier_xgb  1.0000    1.0000     1.0000   1.0000   None    None
     threat_scorer_rf  0.9999    0.9999     0.9999   0.9999   None    None
    threat_scorer_xgb  1.0000    1.0000     1.0000   1.0000   None    None
================================================================================

ATTACK_CLASSIFIER_RF - Detailed Results:
--------------------------------------------------------------------------------
                 precision    recall  f1-score   support

DATABASE_ATTACK       1.00      1.00      1.00       107
   HTTP_EXPLOIT       1.00      1.00      1.00      3255
      PORT_SCAN       1.00      1.00      1.00      1314
     RDP_ATTACK       1.00      1.00      1.00      2144
     SMB_ATTACK       1.00      1.00      1.00       674
SSH_BRUTE_FORCE       1.00      0.99      1.00       189
        TLS_SSL       1.00      1.00      1.00      2269
        UNKNOWN       1.00      1.00      1.00      4114

       accuracy                           1.00     14066
      macro avg       1.00      1.00      1.00     14066
   weighted avg       1.00      1.00      1.00     14066

Top 10 Feature Importances:
                    feature  importance
      protocol_type_encoded    0.300160
             payload_length    0.243745
       payload_type_encoded    0.101098
ip_freq_payload_interaction    0.080853
               ip_frequency    0.073951
```

#### Output Files:

- `models/attack_classifier_rf.pkl` - Random Forest attack classifier (99.99% accuracy)
- `models/attack_classifier_xgb.pkl` - XGBoost attack classifier (100% accuracy)
- `models/threat_scorer_rf.pkl` - Random Forest threat scorer (99.99% accuracy)
- `models/threat_scorer_xgb.pkl` - XGBoost threat scorer (100% accuracy)

#### Training Time:

- Modern laptop: 5-10 seconds total
- Per model: 1-3 seconds
- Training samples: 56,261
- Test samples: 14,066

## Dashboard Usage

### Launching the Dashboard

```powershell
streamlit run app\dashboard.py
```

The dashboard will open automatically at `http://localhost:8501`

### Dashboard Navigation

The dashboard consists of 6 main sections accessible from the sidebar:

#### 1. Overview
- **Purpose**: High-level summary of attack data
- **Features**:
  - Total attack count
  - Unique IP addresses
  - Unique ports targeted
  - Date range coverage
  - Attack timeline chart
  - Attack type distribution
  - Port analysis

#### 2. Attack Analysis
- **Purpose**: Detailed attack pattern analysis
- **Features**:
  - Attack type breakdown
  - Protocol distribution
  - Top attacking IP addresses
  - Most targeted ports
  - Port category analysis

#### 3. Temporal Patterns
- **Purpose**: Time-based attack analysis
- **Features**:
  - Attacks by hour of day
  - Attacks by day of week
  - Weekend vs weekday patterns
  - Attack timeline trends
  - Peak activity identification

#### 4. Threat Intelligence
- **Purpose**: Identify high-risk actors and threats
- **Features**:
  - High-risk IP addresses with reputation scores
  - Severity distribution
  - Severity by attack type
  - Color-coded threat levels
  - Actionable intelligence

#### 5. ML Predictions
- **Purpose**: Machine learning model insights
- **Features**:
  - Attack type classification results
  - Threat severity predictions
  - Model performance metrics
  - Distribution visualizations

#### 6. Advanced Analytics
- **Purpose**: Deep-dive analysis
- **Features**:
  - Payload type analysis
  - IP frequency distribution
  - Advanced statistical metrics
  - Custom analytics

### Dashboard Controls

#### Sidebar Information
- Dataset summary statistics
- Date range
- Server location
- Quick navigation menu

#### Interactive Features
- Hover over charts for detailed information
- Click legend items to show/hide data
- Zoom and pan on time-series charts
- Download charts as PNG images

### Interpreting Results

#### Threat Severity Color Codes

- **Red Background**: HIGH severity threats (immediate action required)
- **Yellow Background**: MEDIUM severity threats (enhanced monitoring)
- **Green Background**: LOW severity threats (standard logging)

#### IP Reputation Scores

- **70-100**: Critical threat (block immediately)
- **40-69**: Moderate threat (rate limit)
- **0-39**: Low threat (monitor)

#### Attack Type Indicators

- **HTTP_EXPLOIT**: Web application attacks
- **DATABASE_ATTACK**: Database exploitation
- **SSH_BRUTE_FORCE**: SSH password attempts
- **TLS_SSL**: SSL/TLS probing
- **RDP_ATTACK**: Remote desktop attacks
- **SMB_ATTACK**: Windows file sharing attacks
- **PORT_SCAN**: Network reconnaissance
- **UNKNOWN**: Unclassified activity

## Troubleshooting

### Common Issues and Solutions

#### Issue 1: "Data file not found"

**Error Message:**
```
FileNotFoundError: dataset\london.csv not found
```

**Solution:**
- Verify the dataset file exists in `dataset/london.csv`
- Check file path and spelling
- Ensure you're in the correct project directory

#### Issue 2: "Import errors" when running scripts

**Error Message:**
```
ModuleNotFoundError: No module named 'pandas'
```

**Solution:**
```powershell
pip install -r requirements.txt
```

#### Issue 3: Dashboard won't load

**Error Message:**
```
Processed data not found. Please run preprocessing first.
```

**Solution:**
Run the complete pipeline in order:
```powershell
python src\data_preprocessing.py
python src\feature_engineering.py
python src\models.py
streamlit run app\dashboard.py
```

#### Issue 4: Low memory warnings

**Solution:**
- Close other applications
- Increase available RAM
- Process data in smaller batches (modify code to sample data)

#### Issue 5: Streamlit port already in use

**Error Message:**
```
Port 8501 is already in use
```

**Solution:**
```powershell
# Use a different port
streamlit run app\dashboard.py --server.port 8502
```

### Performance Optimization

#### Speed Up Processing

For faster development and testing:

1. Use a data sample:
```python
# In data_preprocessing.py, after loading data:
df = df.sample(10000)  # Use 10,000 records instead of all
```

2. Reduce model complexity:
```python
# In config.py, reduce n_estimators:
RF_PARAMS = {
    'n_estimators': 50,  # Reduced from 100
    ...
}
```

#### Reduce Memory Usage

```python
# Use categorical dtypes for memory efficiency
df['port'] = df['port'].astype('category')
df['protocol_type'] = df['protocol_type'].astype('category')
```

### Getting Help

If you encounter issues not covered in this guide:

1. Check the log file: `output/honeypot_intelligence.log`
2. Review error messages for specific details
3. Verify all prerequisites are installed
4. Ensure Python version is 3.9 or higher

### Best Practices

1. **Regular Updates**: Re-run the pipeline when new data is available
2. **Model Monitoring**: Track model performance metrics over time
3. **Data Backup**: Keep backups of original dataset and trained models
4. **Resource Management**: Close dashboard when not in use to free resources
5. **Documentation**: Keep notes on custom modifications or configurations

## Next Steps

After familiarizing yourself with the system:

1. Explore the API Reference for customization options
2. Review the Methodology documentation for ML details
3. Experiment with different model parameters
4. Add new features or attack categories
5. Integrate with external threat intelligence APIs
