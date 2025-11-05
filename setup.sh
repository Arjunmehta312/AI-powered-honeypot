#!/bin/bash
# Setup script for AI-Powered Honeypot Intelligence System (Linux/Mac)

echo "========================================"
echo "AI-Powered Honeypot Intelligence System"
echo "Setup Script"
echo "========================================"
echo ""

# Check Python version
echo "Checking Python version..."
python3 --version
if [ $? -ne 0 ]; then
    echo "ERROR: Python 3 is not installed!"
    echo "Please install Python 3.9 or higher"
    exit 1
fi

# Create virtual environment
echo ""
echo "Creating virtual environment..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "ERROR: Failed to create virtual environment!"
    exit 1
fi

# Activate virtual environment
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages
echo ""
echo "Installing required packages..."
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn streamlit plotly

if [ $? -ne 0 ]; then
    echo ""
    echo "ERROR: Package installation failed!"
    exit 1
fi

# Create output directory if it doesn't exist
mkdir -p output
mkdir -p models

echo ""
echo "========================================"
echo "Setup completed successfully!"
echo "========================================"
echo ""
echo "To activate the virtual environment, run:"
echo "    source venv/bin/activate"
echo ""
echo "To run the pipeline, execute:"
echo "    python src/data_preprocessing.py"
echo "    python src/feature_engineering.py"
echo "    python src/models.py"
echo "    streamlit run app/dashboard.py"
echo ""
