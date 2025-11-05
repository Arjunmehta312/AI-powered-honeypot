"""
System Status Page - Check if all required files and models are available
"""

import streamlit as st
from pathlib import Path
import pandas as pd

st.set_page_config(
    page_title="System Status",
    page_icon="✅",
    layout="wide"
)

st.title("🔍 System Status Check")
st.write("This page verifies all required files and dependencies are available.")

# Check paths
project_root = Path(__file__).parent.parent
dataset_path = project_root / 'dataset' / 'london.csv'
processed_path = project_root / 'output' / 'processed_data.csv'
features_path = project_root / 'output' / 'feature_data.csv'
models_path = project_root / 'models'

st.header("📂 File System Check")

files_to_check = {
    "Raw Dataset": dataset_path,
    "Processed Data": processed_path,
    "Feature Data": features_path,
    "Attack Classifier Model": models_path / 'attack_classifier.pkl',
    "Threat Scorer Model": models_path / 'threat_scorer.pkl',
    "Feature Columns": models_path / 'feature_columns.pkl',
    "Label Encoder": models_path / 'label_encoder.pkl',
}

col1, col2 = st.columns(2)

for name, path in files_to_check.items():
    with col1:
        st.write(f"**{name}:**")
    with col2:
        if path.exists():
            st.success(f"✅ Found ({path.stat().st_size / 1024 / 1024:.2f} MB)")
        else:
            st.error(f"❌ Not found")
            st.caption(f"Expected: `{path}`")

st.header("📊 Data Preview")

if processed_path.exists():
    try:
        df = pd.read_csv(processed_path, nrows=5)
        st.success(f"✅ Loaded {len(df)} sample records from processed data")
        st.dataframe(df.head())
        
        # Full stats
        df_full = pd.read_csv(processed_path)
        st.metric("Total Records", f"{len(df_full):,}")
        st.metric("Columns", len(df_full.columns))
        st.write("**Columns:**", ", ".join(df_full.columns.tolist()))
    except Exception as e:
        st.error(f"Error loading data: {e}")
else:
    st.warning("Processed data file not found")

st.header("🐍 Python Environment")

import sys
import platform

st.write(f"**Python Version:** {platform.python_version()}")
st.write(f"**Platform:** {platform.platform()}")
st.write(f"**Working Directory:** `{Path.cwd()}`")

st.header("📦 Installed Packages")

try:
    import pandas
    import numpy
    import sklearn
    import xgboost
    import plotly
    import joblib
    
    packages = {
        "pandas": pandas.__version__,
        "numpy": numpy.__version__,
        "scikit-learn": sklearn.__version__,
        "xgboost": xgboost.__version__,
        "plotly": plotly.__version__,
        "joblib": joblib.__version__,
    }
    
    for pkg, ver in packages.items():
        st.success(f"✅ {pkg}: {ver}")
except ImportError as e:
    st.error(f"Missing package: {e}")

st.header("🔧 Quick Fix")

if not processed_path.exists() or not features_path.exists():
    st.warning("Missing data files! Click below to generate them:")
    
    if st.button("🚀 Run Preprocessing & Feature Engineering"):
        with st.spinner("Running preprocessing..."):
            try:
                # Import and run preprocessing
                import sys
                sys.path.insert(0, str(project_root / 'src'))
                from data_preprocessing import main as preprocess_main
                from feature_engineering import main as feature_main
                
                preprocess_main()
                st.success("✅ Preprocessing complete!")
                
                feature_main()
                st.success("✅ Feature engineering complete!")
                
                st.rerun()
            except Exception as e:
                st.error(f"Error: {e}")
                st.exception(e)
