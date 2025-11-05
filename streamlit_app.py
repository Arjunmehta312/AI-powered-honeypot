"""
AI-Powered Honeypot Intelligence Dashboard
Entry point for Streamlit Cloud deployment
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the main dashboard
from app.dashboard import main

if __name__ == "__main__":
    main()
