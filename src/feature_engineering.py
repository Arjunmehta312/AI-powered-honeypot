"""
Feature Engineering Module for Honeypot Intelligence System.

This module creates advanced features and prepares data for machine learning models.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, List, Optional
from pathlib import Path
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib

from config import (
    PROCESSED_DATA_FILE, FEATURES_FILE, LABEL_ENCODER_FILE,
    FEATURE_SCALER_FILE, NUMERICAL_FEATURES, CATEGORICAL_FEATURES,
    ATTACK_TYPES, SEVERITY_LEVELS, LOG_FORMAT, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class FeatureEngineer:
    """
    Handles feature engineering and transformation for ML models.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the FeatureEngineer.
        
        Args:
            data_path: Path to the preprocessed data file.
        """
        self.data_path = data_path or PROCESSED_DATA_FILE
        self.df = None
        self.label_encoders = {}
        self.scaler = None
        
    def load_processed_data(self) -> pd.DataFrame:
        """
        Load preprocessed data.
        
        Returns:
            DataFrame containing preprocessed data.
        """
        logger.info(f"Loading processed data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
            return self.df
        except FileNotFoundError:
            logger.error(f"Processed data file not found: {self.data_path}")
            raise
    
    def create_attack_labels(self) -> pd.DataFrame:
        """
        Create attack type labels based on protocol and payload characteristics.
        
        Returns:
            DataFrame with attack_type label.
        """
        logger.info("Creating attack labels...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        def classify_attack(row):
            protocol = row['protocol_type']
            port = row['port']
            payload_length = row['payload_length']
            
            # TLS/SSL attacks
            if protocol == 'TLS_SSL':
                return 'TLS_SSL'
            
            # HTTP exploits
            if protocol == 'HTTP':
                return 'HTTP_EXPLOIT'
            
            # SSH brute force
            if protocol == 'SSH':
                return 'SSH_BRUTE_FORCE'
            
            # SMB attacks
            if protocol == 'SMB':
                return 'SMB_ATTACK'
            
            # RDP attacks
            if protocol == 'RDP':
                return 'RDP_ATTACK'
            
            # Database attacks
            if protocol in ['REDIS', 'MYSQL', 'MONGODB'] or port in [3306, 6379, 27017]:
                return 'DATABASE_ATTACK'
            
            # Port scanning (empty payload, low length)
            if payload_length < 10 or protocol == 'EMPTY':
                return 'PORT_SCAN'
            
            return 'UNKNOWN'
        
        self.df['attack_type'] = self.df.apply(classify_attack, axis=1)
        
        # Log distribution
        attack_dist = self.df['attack_type'].value_counts()
        logger.info(f"Attack type distribution:\n{attack_dist}")
        
        return self.df
    
    def create_severity_labels(self) -> pd.DataFrame:
        """
        Create threat severity labels based on attack characteristics.
        
        Returns:
            DataFrame with severity label.
        """
        logger.info("Creating severity labels...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        def calculate_severity(row):
            attack_type = row['attack_type']
            ip_freq = row['ip_frequency']
            has_malicious = row['has_malicious_keyword']
            payload_length = row['payload_length']
            
            # High severity attacks
            high_severity_attacks = ['HTTP_EXPLOIT', 'DATABASE_ATTACK', 'SMB_ATTACK']
            if attack_type in high_severity_attacks:
                return 'HIGH'
            
            # Repeat offenders
            if ip_freq > 50:
                return 'HIGH'
            
            # Malicious keywords present
            if has_malicious and payload_length > 100:
                return 'HIGH'
            
            # Medium severity
            medium_severity_attacks = ['SSH_BRUTE_FORCE', 'RDP_ATTACK', 'TLS_SSL']
            if attack_type in medium_severity_attacks:
                return 'MEDIUM'
            
            if ip_freq > 10:
                return 'MEDIUM'
            
            # Low severity (port scans, unknown)
            return 'LOW'
        
        self.df['severity'] = self.df.apply(calculate_severity, axis=1)
        
        # Log distribution
        severity_dist = self.df['severity'].value_counts()
        logger.info(f"Severity distribution:\n{severity_dist}")
        
        return self.df
    
    def create_ip_reputation_score(self) -> pd.DataFrame:
        """
        Create IP reputation score based on attack patterns.
        
        Returns:
            DataFrame with ip_reputation_score.
        """
        logger.info("Creating IP reputation scores...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        # Calculate score components
        ip_stats = self.df.groupby('from').agg({
            'attack_type': 'count',  # Total attacks
            'severity': lambda x: (x == 'HIGH').sum(),  # High severity count
            'has_malicious_keyword': 'sum',  # Malicious keyword count
            'payload_length': 'mean'  # Average payload length
        }).reset_index()
        
        ip_stats.columns = ['from', 'total_attacks', 'high_severity_attacks', 
                            'malicious_count', 'avg_payload_length']
        
        # Normalize and calculate score (0-100)
        ip_stats['reputation_score'] = (
            (ip_stats['total_attacks'] / ip_stats['total_attacks'].max() * 30) +
            (ip_stats['high_severity_attacks'] / ip_stats['total_attacks'].max() * 40) +
            (ip_stats['malicious_count'] / ip_stats['total_attacks'].max() * 30)
        )
        
        ip_stats['reputation_score'] = ip_stats['reputation_score'].clip(0, 100).round(2)
        
        # Merge back to main dataframe
        self.df = self.df.merge(
            ip_stats[['from', 'reputation_score']], 
            on='from', 
            how='left'
        )
        
        logger.info("IP reputation scores created")
        
        return self.df
    
    def encode_categorical_features(self) -> pd.DataFrame:
        """
        Encode categorical features using LabelEncoder.
        
        Returns:
            DataFrame with encoded categorical features.
        """
        logger.info("Encoding categorical features...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        categorical_cols = ['payload_type', 'protocol_type', 'port_category']
        
        for col in categorical_cols:
            if col in self.df.columns:
                le = LabelEncoder()
                self.df[f'{col}_encoded'] = le.fit_transform(self.df[col].astype(str))
                self.label_encoders[col] = le
                logger.info(f"Encoded {col}: {len(le.classes_)} unique values")
        
        return self.df
    
    def create_interaction_features(self) -> pd.DataFrame:
        """
        Create interaction features between important variables.
        
        Returns:
            DataFrame with interaction features.
        """
        logger.info("Creating interaction features...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        # Port and time interaction
        self.df['port_hour_interaction'] = self.df['port'] * self.df['hour']
        
        # IP frequency and payload length
        self.df['ip_freq_payload_interaction'] = (
            self.df['ip_frequency'] * np.log1p(self.df['payload_length'])
        )
        
        # Weekend and night combination
        self.df['weekend_night'] = self.df['is_weekend'] * self.df['is_night']
        
        logger.info("Interaction features created")
        
        return self.df
    
    def create_time_based_features(self) -> pd.DataFrame:
        """
        Create advanced time-based features.
        
        Returns:
            DataFrame with time-based features.
        """
        logger.info("Creating time-based features...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        # Parse timestamp if not already done
        if 'timestamp' not in self.df.columns:
            self.df['timestamp'] = pd.to_datetime(self.df['time'], format='%m/%d/%Y, %H:%M:%S')
        else:
            self.df['timestamp'] = pd.to_datetime(self.df['timestamp'])
        
        # Sort by timestamp
        self.df = self.df.sort_values('timestamp').reset_index(drop=True)
        
        # Time since previous attack (in seconds)
        self.df['time_since_prev_attack'] = (
            self.df['timestamp'].diff().dt.total_seconds().fillna(0)
        )
        
        # Attacks per hour window
        self.df['attacks_per_hour'] = self.df.groupby(
            pd.Grouper(key='timestamp', freq='H')
        )['timestamp'].transform('count')
        
        logger.info("Time-based features created")
        
        return self.df
    
    def prepare_features_for_modeling(self) -> Tuple[pd.DataFrame, List[str]]:
        """
        Prepare final feature set for modeling.
        
        Returns:
            Tuple of (features DataFrame, list of feature names)
        """
        logger.info("Preparing features for modeling...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        # Define feature columns
        feature_columns = [
            # Numerical features
            'payload_length',
            'port',
            'hour',
            'day_of_week',
            'day',
            'is_weekend',
            'is_night',
            'ip_frequency',
            'port_frequency',
            'has_malicious_keyword',
            'reputation_score',
            
            # Encoded categorical features
            'payload_type_encoded',
            'protocol_type_encoded',
            'port_category_encoded',
            
            # Interaction features
            'port_hour_interaction',
            'ip_freq_payload_interaction',
            'weekend_night',
            
            # Time-based features
            'time_since_prev_attack',
            'attacks_per_hour'
        ]
        
        # Filter to existing columns
        available_features = [col for col in feature_columns if col in self.df.columns]
        
        logger.info(f"Selected {len(available_features)} features for modeling")
        
        return self.df, available_features
    
    def scale_features(self, feature_columns: List[str]) -> pd.DataFrame:
        """
        Scale numerical features using StandardScaler.
        
        Args:
            feature_columns: List of feature column names to scale.
            
        Returns:
            DataFrame with scaled features.
        """
        logger.info("Scaling features...")
        
        if self.df is None:
            raise ValueError("Data not loaded. Call load_processed_data() first.")
        
        # Select only numerical features for scaling
        numerical_cols = [col for col in feature_columns 
                         if self.df[col].dtype in ['int64', 'float64']]
        
        self.scaler = StandardScaler()
        self.df[numerical_cols] = self.scaler.fit_transform(self.df[numerical_cols])
        
        logger.info(f"Scaled {len(numerical_cols)} numerical features")
        
        return self.df
    
    def engineer_features(self, scale: bool = False) -> Tuple[pd.DataFrame, List[str]]:
        """
        Execute full feature engineering pipeline.
        
        Args:
            scale: Whether to scale features.
            
        Returns:
            Tuple of (engineered DataFrame, list of feature names)
        """
        logger.info("Starting full feature engineering pipeline...")
        
        self.load_processed_data()
        self.create_attack_labels()
        self.create_severity_labels()
        self.create_ip_reputation_score()
        self.encode_categorical_features()
        self.create_interaction_features()
        self.create_time_based_features()
        
        df_features, feature_columns = self.prepare_features_for_modeling()
        
        if scale:
            df_features = self.scale_features(feature_columns)
        
        logger.info("Feature engineering pipeline complete")
        
        return df_features, feature_columns
    
    def save_engineered_data(self, output_path: Optional[Path] = None) -> None:
        """
        Save engineered features to CSV file.
        
        Args:
            output_path: Path to save the features. Defaults to FEATURES_FILE.
        """
        if self.df is None:
            raise ValueError("No data to save. Run feature engineering first.")
        
        output_path = output_path or FEATURES_FILE
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        self.df.to_csv(output_path, index=False)
        logger.info(f"Engineered features saved to {output_path}")
    
    def save_encoders(self) -> None:
        """Save label encoders and scaler for future use."""
        LABEL_ENCODER_FILE.parent.mkdir(parents=True, exist_ok=True)
        
        if self.label_encoders:
            joblib.dump(self.label_encoders, LABEL_ENCODER_FILE)
            logger.info(f"Label encoders saved to {LABEL_ENCODER_FILE}")
        
        if self.scaler:
            joblib.dump(self.scaler, FEATURE_SCALER_FILE)
            logger.info(f"Feature scaler saved to {FEATURE_SCALER_FILE}")


def main():
    """Main function to run feature engineering pipeline."""
    engineer = FeatureEngineer()
    df_features, feature_columns = engineer.engineer_features(scale=False)
    engineer.save_engineered_data()
    engineer.save_encoders()
    
    print(f"\nFeature Engineering Complete!")
    print(f"Total Features: {len(feature_columns)}")
    print(f"\nFeature Columns:")
    for i, col in enumerate(feature_columns, 1):
        print(f"  {i}. {col}")
    print(f"\nEngineered data saved to: {FEATURES_FILE}")


if __name__ == "__main__":
    main()
