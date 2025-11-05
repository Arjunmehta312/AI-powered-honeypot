"""
Data Preprocessing Module for Honeypot Intelligence System.

This module handles loading, cleaning, and initial processing of honeypot attack data.
"""

import pandas as pd
import numpy as np
import re
import logging
from datetime import datetime
from typing import Tuple, Optional
from pathlib import Path

from config import (
    LONDON_DATA_FILE, PROCESSED_DATA_FILE, RANDOM_STATE,
    LOG_FORMAT, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class DataPreprocessor:
    """
    Handles data loading, cleaning, and preprocessing for honeypot attack logs.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the DataPreprocessor.
        
        Args:
            data_path: Path to the CSV data file. Defaults to LONDON_DATA_FILE.
        """
        self.data_path = data_path or LONDON_DATA_FILE
        self.df = None
        
    def load_data(self) -> pd.DataFrame:
        """
        Load honeypot attack data from CSV file.
        
        Returns:
            DataFrame containing raw attack data.
        """
        logger.info(f"Loading data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
            return self.df
        except FileNotFoundError:
            logger.error(f"Data file not found: {self.data_path}")
            raise
        except Exception as e:
            logger.error(f"Error loading data: {str(e)}")
            raise
    
    def clean_data(self) -> pd.DataFrame:
        """
        Clean the loaded data by handling missing values and formatting issues.
        
        Returns:
            Cleaned DataFrame.
        """
        logger.info("Cleaning data...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Create a copy to avoid modifying original
        df_clean = self.df.copy()
        
        # Remove duplicate records
        initial_count = len(df_clean)
        df_clean = df_clean.drop_duplicates()
        duplicates_removed = initial_count - len(df_clean)
        if duplicates_removed > 0:
            logger.info(f"Removed {duplicates_removed} duplicate records")
        
        # Clean IP addresses (remove quotes)
        df_clean['from'] = df_clean['from'].astype(str).str.strip("'\"")
        
        # Clean port values
        df_clean['port'] = pd.to_numeric(df_clean['port'], errors='coerce')
        
        # Clean payload values (handle null/empty)
        df_clean['payload'] = df_clean['payload'].fillna('')
        
        # Handle missing values
        missing_before = df_clean.isnull().sum().sum()
        if missing_before > 0:
            logger.warning(f"Found {missing_before} missing values")
            # Drop rows with missing critical fields
            df_clean = df_clean.dropna(subset=['time', 'from', 'port'])
            logger.info(f"Dropped rows with missing critical fields")
        
        # Clean country field
        df_clean['country'] = df_clean['country'].astype(str).str.strip()
        
        self.df = df_clean
        logger.info(f"Cleaning complete. {len(df_clean)} records remaining")
        
        return df_clean
    
    def parse_timestamps(self) -> pd.DataFrame:
        """
        Parse and extract datetime features from timestamp field.
        
        Returns:
            DataFrame with parsed datetime features.
        """
        logger.info("Parsing timestamps...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Parse time field to datetime
        self.df['timestamp'] = pd.to_datetime(self.df['time'], format='%m/%d/%Y, %H:%M:%S')
        
        # Extract temporal features
        self.df['hour'] = self.df['timestamp'].dt.hour
        self.df['day'] = self.df['timestamp'].dt.day
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month'] = self.df['timestamp'].dt.month
        self.df['date'] = self.df['timestamp'].dt.date
        
        # Create derived temporal features
        self.df['is_weekend'] = (self.df['day_of_week'] >= 5).astype(int)
        self.df['is_night'] = ((self.df['hour'] >= 22) | (self.df['hour'] <= 6)).astype(int)
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        logger.info("Timestamp parsing complete")
        
        return self.df
    
    def extract_payload_features(self) -> pd.DataFrame:
        """
        Extract features from payload field.
        
        Returns:
            DataFrame with payload features.
        """
        logger.info("Extracting payload features...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # Payload length
        self.df['payload_length'] = self.df['payload'].astype(str).apply(len)
        
        # Payload type (binary, text, empty)
        def categorize_payload(payload):
            if not payload or payload == '' or payload == "''":
                return 'EMPTY'
            elif payload.startswith("b'") or payload.startswith('b"'):
                return 'BINARY'
            else:
                return 'TEXT'
        
        self.df['payload_type'] = self.df['payload'].apply(categorize_payload)
        
        # Check for malicious keywords
        malicious_patterns = [
            'GET', 'POST', 'admin', 'manager', 'script',
            'exploit', 'shell', 'cmd', 'exec', 'eval'
        ]
        
        def has_malicious_keyword(payload):
            payload_str = str(payload).lower()
            return any(keyword.lower() in payload_str for keyword in malicious_patterns)
        
        self.df['has_malicious_keyword'] = self.df['payload'].apply(has_malicious_keyword).astype(int)
        
        logger.info("Payload feature extraction complete")
        
        return self.df
    
    def calculate_frequency_features(self) -> pd.DataFrame:
        """
        Calculate frequency-based features (IP frequency, port frequency).
        
        Returns:
            DataFrame with frequency features.
        """
        logger.info("Calculating frequency features...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        # IP address frequency
        ip_counts = self.df['from'].value_counts()
        self.df['ip_frequency'] = self.df['from'].map(ip_counts)
        
        # Port frequency
        port_counts = self.df['port'].value_counts()
        self.df['port_frequency'] = self.df['port'].map(port_counts)
        
        # Port category
        def categorize_port(port):
            if pd.isna(port):
                return 'UNKNOWN'
            port = int(port)
            if port < 1024:
                return 'WELL_KNOWN'
            elif port < 49152:
                return 'REGISTERED'
            else:
                return 'DYNAMIC'
        
        self.df['port_category'] = self.df['port'].apply(categorize_port)
        
        logger.info("Frequency feature calculation complete")
        
        return self.df
    
    def identify_protocol(self) -> pd.DataFrame:
        """
        Identify protocol type from payload patterns.
        
        Returns:
            DataFrame with protocol identification.
        """
        logger.info("Identifying protocols...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        def detect_protocol(payload):
            payload_str = str(payload)
            
            # Check for TLS/SSL
            if '\\x16\\x03' in payload_str or 'SSL' in payload_str or 'TLS' in payload_str:
                return 'TLS_SSL'
            
            # Check for HTTP
            if any(x in payload_str for x in ['GET ', 'POST ', 'PUT ', 'HTTP/1.']):
                return 'HTTP'
            
            # Check for SSH
            if 'SSH-2.0' in payload_str or 'SSH-1.' in payload_str:
                return 'SSH'
            
            # Check for SMB
            if 'SMB' in payload_str or '\\xffSMB' in payload_str:
                return 'SMB'
            
            # Check for RDP
            if 'mstshash' in payload_str or '\\x03\\x00\\x00' in payload_str:
                return 'RDP'
            
            # Check for Redis
            if 'info\\r\\n' in payload_str or 'redis' in payload_str.lower():
                return 'REDIS'
            
            # Check for MySQL
            if 'mysql' in payload_str.lower():
                return 'MYSQL'
            
            # Check for SIP
            if 'SIP/2.0' in payload_str or 'sip:' in payload_str:
                return 'SIP'
            
            # Check for ADB
            if 'CNXN' in payload_str:
                return 'ADB'
            
            # Empty payload
            if not payload_str or payload_str == '' or payload_str == "''":
                return 'EMPTY'
            
            return 'UNKNOWN'
        
        self.df['protocol_type'] = self.df['payload'].apply(detect_protocol)
        
        logger.info("Protocol identification complete")
        
        return self.df
    
    def preprocess(self) -> pd.DataFrame:
        """
        Execute full preprocessing pipeline.
        
        Returns:
            Fully preprocessed DataFrame.
        """
        logger.info("Starting full preprocessing pipeline...")
        
        self.load_data()
        self.clean_data()
        self.parse_timestamps()
        self.extract_payload_features()
        self.calculate_frequency_features()
        self.identify_protocol()
        
        logger.info("Preprocessing pipeline complete")
        
        return self.df
    
    def save_processed_data(self, output_path: Optional[Path] = None) -> None:
        """
        Save preprocessed data to CSV file.
        
        Args:
            output_path: Path to save the processed data. Defaults to PROCESSED_DATA_FILE.
        """
        if self.df is None:
            raise ValueError("No data to save. Run preprocessing first.")
        
        output_path = output_path or PROCESSED_DATA_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Processed data saved to {output_path}")
    
    def get_data_summary(self) -> dict:
        """
        Get summary statistics of the processed data.
        
        Returns:
            Dictionary containing summary statistics.
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")
        
        summary = {
            'total_records': len(self.df),
            'unique_ips': self.df['from'].nunique(),
            'unique_ports': self.df['port'].nunique(),
            'date_range': {
                'start': self.df['timestamp'].min(),
                'end': self.df['timestamp'].max()
            },
            'protocol_distribution': self.df['protocol_type'].value_counts().to_dict(),
            'payload_type_distribution': self.df['payload_type'].value_counts().to_dict(),
            'country_distribution': self.df['country'].value_counts().to_dict()
        }
        
        return summary


def main():
    """Main function to run preprocessing pipeline."""
    preprocessor = DataPreprocessor()
    df = preprocessor.preprocess()
    preprocessor.save_processed_data()
    
    summary = preprocessor.get_data_summary()
    logger.info(f"Data Summary: {summary}")
    
    print(f"\nPreprocessing Complete!")
    print(f"Total Records: {summary['total_records']}")
    print(f"Unique IPs: {summary['unique_ips']}")
    print(f"Unique Ports: {summary['unique_ports']}")
    print(f"\nProcessed data saved to: {PROCESSED_DATA_FILE}")


if __name__ == "__main__":
    main()
