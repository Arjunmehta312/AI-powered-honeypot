# AI-Powered Honeypot Intelligence System

![Python](https://img.shields.io/badge/python-3.9+-blue.svg)
![License](https://img.shields.io/badge/license-MIT-green.svg)
![Accuracy](https://img.shields.io/badge/accuracy-100%25-brightgreen.svg)
![Status](https://img.shields.io/badge/status-active-success.svg)

A comprehensive machine learning solution for analyzing, classifying, and predicting cyber attacks from honeypot server logs.

## Overview

This system processes attack logs from AWS EC2 honeypot servers to provide automated threat classification, severity assessment, and actionable intelligence. Built with Python and powered by Random Forest and XGBoost algorithms, it delivers **99.99-100% classification accuracy** for identifying and categorizing cyber threats.

## Key Features

- Automated classification of 8 attack types with 100% accuracy
- Threat severity assessment (LOW, MEDIUM, HIGH) with 99.99% accuracy
- IP reputation scoring system (0-100 scale)
- Interactive web dashboard with real-time analytics
- Temporal and behavioral pattern analysis
- 70,327 attack records processed from 3,508 unique IPs
- 19 engineered features including behavioral patterns
- CPU-optimized algorithms (no GPU required)
- Protocol identification for 10 network protocols

## Project Structure

```
Project/
├── dataset/              # Raw honeypot data
│   └── london.csv        # Attack logs (70K+ records)
├── src/                  # Source code modules
│   ├── config.py         # Configuration settings
│   ├── data_preprocessing.py   # Data cleaning pipeline
│   ├── feature_engineering.py  # Feature creation
│   └── models.py         # ML model training
├── app/                  # Streamlit dashboard
│   └── dashboard.py      # Interactive web interface
├── models/               # Trained ML models (generated)
│   ├── attack_classifier_rf.pkl
│   ├── attack_classifier_xgb.pkl
│   ├── threat_scorer_rf.pkl
│   └── threat_scorer_xgb.pkl
├── output/               # Processed data (generated)
│   ├── processed_data.csv
│   └── feature_data.csv
├── docs/                 # Documentation
│   ├── README.md         # System architecture
│   ├── USER_GUIDE.md     # User guide
│   └── METHODOLOGY.md    # ML methodology
├── requirements.txt      # Python dependencies
├── .gitignore           # Git ignore rules
├── LICENSE              # MIT license
└── README.md            # This file
```

## Quick Start

### 1. Clone and Setup

```bash
# Clone the repository
git clone <repository-url>
cd Project

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Install dependencies
pip install numpy pandas scikit-learn xgboost joblib matplotlib seaborn streamlit plotly
```

### 2. Run Pipeline

**Automated (Windows):**
```cmd
run_pipeline.bat
```

**Manual (All platforms):**
```bash
# Step 1: Preprocess data
python src/data_preprocessing.py

# Step 2: Engineer features
python src/feature_engineering.py

# Step 3: Train models
python src/models.py

# Step 4: Launch dashboard
streamlit run app/dashboard.py
```

### 3. Access Dashboard

Open browser to: **http://localhost:8501**

## System Requirements

- Python 3.9+
- 4GB RAM minimum
- 2GB storage
- No GPU required

## Performance

- Data Processing: 3-5 seconds (70,327 records)
- Feature Engineering: 1-2 seconds (19 features)
- Model Training: 5-10 seconds (4 models)
- Prediction: <1ms per sample
- Dashboard Load: 2-3 seconds

## Model Accuracy

**Attack Type Classification:**
- Random Forest: **99.99% accuracy**
- XGBoost: **100% accuracy**

**Threat Severity Scoring:**
- Random Forest: **99.99% accuracy**
- XGBoost: **100% accuracy**

**Training Details:**
- Training samples: 56,261
- Test samples: 14,066
- Features: 19
- Classes: 8 attack types, 3 severity levels

## Attack Categories

1. **TLS_SSL** (11,344) - TLS/SSL handshake attempts
2. **HTTP_EXPLOIT** (16,272) - Web application exploits
3. **SSH_BRUTE_FORCE** (947) - SSH authentication attacks
4. **SMB_ATTACK** (3,368) - SMB protocol exploits
5. **RDP_ATTACK** (10,721) - Remote desktop attacks
6. **DATABASE_ATTACK** (536) - Database exploitation (Redis, MySQL)
7. **PORT_SCAN** (6,571) - Network reconnaissance
8. **UNKNOWN** (20,568) - Unclassified attacks

## Technology Stack

- **Language**: Python 3.9+
- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn 1.7.2, xgboost 1.7.6
- **Visualization**: plotly, matplotlib, seaborn
- **Dashboard**: Streamlit 1.51.0
- **Model Persistence**: joblib

## Documentation

Full documentation available in `/docs`:
- [System Architecture](docs/README.md) - Overview and architecture
- [User Guide](docs/USER_GUIDE.md) - Installation and usage instructions
- [Methodology](docs/METHODOLOGY.md) - ML methodology and algorithms
- [Quick Start](QUICKSTART.md) - Get started in 3 minutes
- [Contributing](CONTRIBUTING.md) - Contribution guidelines

1. TLS_SSL - TLS/SSL handshake attempts
2. HTTP_EXPLOIT - Web application exploits
3. SSH_BRUTE_FORCE - SSH authentication attacks
4. SMB_ATTACK - SMB protocol exploits
5. RDP_ATTACK - Remote desktop attacks
6. DATABASE_ATTACK - Database exploitation
7. PORT_SCAN - Network reconnaissance
8. UNKNOWN - Unclassified attacks

## Technology Stack

- **Data Processing**: pandas, numpy
- **Machine Learning**: scikit-learn, xgboost
- **Visualization**: plotly, matplotlib, seaborn
- **Dashboard**: Streamlit
- **Notebooks**: Jupyter

## Dataset

- Source: AWS EC2 Honeypot (London)
- Period: May 1-5, 2021
- Total Records: 70,327 attacks (after cleaning)
- Unique IPs: 3,508 attackers
- Unique Ports: 30,337 ports targeted
- Fields: timestamp, payload, source IP, port, country
- Protocols Identified: 10 (TLS_SSL, HTTP, RDP, SMB, SSH, REDIS, SIP, ADB, MYSQL, UNKNOWN)

## Engineered Features (19)

1. payload_length - Size of attack payload
2. port - Target port number
3. hour - Hour of attack (0-23)
4. day_of_week - Day of week (0-6)
5. day - Day of month
6. is_weekend - Weekend indicator
7. is_night - Night-time indicator (22:00-06:00)
8. ip_frequency - Attack frequency per IP
9. port_frequency - Port targeting frequency
10. has_malicious_keyword - Malicious pattern detection
11. reputation_score - IP reputation (0-100)
12. payload_type_encoded - Payload type category
13. protocol_type_encoded - Protocol category
14. port_category_encoded - Port category
15. port_hour_interaction - Port-time interaction
16. ip_freq_payload_interaction - IP-payload interaction
17. weekend_night - Weekend night attacks
18. time_since_prev_attack - Time delta between attacks
19. attacks_per_hour - Hourly attack rate

## Screenshots

### Dashboard Overview
Interactive web interface showing attack statistics, temporal patterns, and ML predictions.

### Model Performance
Perfect 100% accuracy achieved with XGBoost classifiers on real-world honeypot data.

## Future Enhancements

- [ ] Real-time attack detection and alerting
- [ ] Integration with SIEM systems
- [ ] Deep learning models (LSTM for sequence analysis)
- [ ] Automated response mechanisms
- [ ] Cloud deployment (AWS/Azure/GCP)
- [ ] API endpoint for predictions
- [ ] Email/SMS alert notifications

## Contributors

AI/ML Course Project - 2025

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Dataset: AWS EC2 Honeypot (London)
- Libraries: scikit-learn, XGBoost, Streamlit, Plotly
- Course: Artificial Intelligence and Machine Learning

## Contact

For questions or issues, please open a GitHub issue.
