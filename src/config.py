"""
Configuration settings for the Honeypot Intelligence System.
"""

import os
from pathlib import Path

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "dataset"
MODELS_DIR = PROJECT_ROOT / "models"
OUTPUT_DIR = PROJECT_ROOT / "output"
DOCS_DIR = PROJECT_ROOT / "docs"

# Data files
LONDON_DATA_FILE = DATA_DIR / "london.csv"
PROCESSED_DATA_FILE = OUTPUT_DIR / "processed_data.csv"
FEATURES_FILE = OUTPUT_DIR / "feature_data.csv"

# Model files
ATTACK_CLASSIFIER_MODEL = MODELS_DIR / "attack_classifier.pkl"
THREAT_SCORER_MODEL = MODELS_DIR / "threat_scorer.pkl"
LABEL_ENCODER_FILE = MODELS_DIR / "label_encoder.pkl"
FEATURE_SCALER_FILE = MODELS_DIR / "feature_scaler.pkl"

# Model parameters
RANDOM_STATE = 42
TEST_SIZE = 0.2
VALIDATION_SIZE = 0.1

# Random Forest parameters
RF_PARAMS = {
    'n_estimators': 100,
    'max_depth': 20,
    'min_samples_split': 5,
    'min_samples_leaf': 2,
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'class_weight': 'balanced'
}

# XGBoost parameters
XGB_PARAMS = {
    'n_estimators': 100,
    'max_depth': 6,
    'learning_rate': 0.1,
    'tree_method': 'hist',
    'n_jobs': -1,
    'random_state': RANDOM_STATE,
    'eval_metric': 'mlogloss'
}

# Feature engineering settings
NUMERICAL_FEATURES = [
    'payload_length',
    'port',
    'hour',
    'day_of_week',
    'day',
    'is_weekend',
    'is_night',
    'ip_frequency',
    'port_frequency'
]

CATEGORICAL_FEATURES = [
    'payload_type',
    'protocol_type',
    'port_category'
]

# Attack type categories
ATTACK_TYPES = [
    'TLS_SSL',
    'HTTP_EXPLOIT',
    'SSH_BRUTE_FORCE',
    'SMB_ATTACK',
    'RDP_ATTACK',
    'DATABASE_ATTACK',
    'PORT_SCAN',
    'UNKNOWN'
]

# Threat severity levels
SEVERITY_LEVELS = ['LOW', 'MEDIUM', 'HIGH']

# Port categories
WELL_KNOWN_PORTS = range(0, 1024)
REGISTERED_PORTS = range(1024, 49152)
DYNAMIC_PORTS = range(49152, 65536)

# Protocol signatures
PROTOCOL_SIGNATURES = {
    'TLS_SSL': [b'\\x16\\x03', 'SSL', 'TLS'],
    'HTTP': ['GET ', 'POST ', 'PUT ', 'DELETE ', 'HTTP/1.'],
    'SSH': ['SSH-2.0', 'SSH-1.'],
    'SMB': [b'\\xffSMB', 'SMB'],
    'RDP': [b'\\x03\\x00\\x00', 'Cookie: mstshash', 'RDP'],
    'REDIS': ['*1\\r\\n$4\\r\\ninfo'],
    'MYSQL': ['mysql', 'MySQL'],
    'MONGODB': ['mongo'],
    'SIP': ['SIP/2.0', 'INVITE sip:', 'OPTIONS sip:'],
    'ADB': [b'CNXN']
}

# Malicious keywords
MALICIOUS_KEYWORDS = [
    'exploit', 'admin', 'manager', 'phpmyadmin', 
    'wp-admin', 'script', 'shell', 'cmd', 'exec',
    'eval', 'base64', '../', '/etc/passwd'
]

# Dashboard settings
DASHBOARD_TITLE = "AI-Powered Honeypot Intelligence System"
DASHBOARD_ICON = "shield"
REFRESH_INTERVAL = 5000  # milliseconds

# Visualization settings
PLOT_STYLE = 'seaborn'
FIGURE_DPI = 100
COLOR_PALETTE = 'viridis'

# API endpoints (for future integration)
ABUSEIPDB_API_KEY = os.getenv('ABUSEIPDB_API_KEY', '')
VIRUSTOTAL_API_KEY = os.getenv('VIRUSTOTAL_API_KEY', '')

# Logging configuration
LOG_LEVEL = 'INFO'
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_FILE = OUTPUT_DIR / 'honeypot_intelligence.log'
