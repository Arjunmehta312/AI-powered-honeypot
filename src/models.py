"""
Machine Learning Models Module for Honeypot Intelligence System.

This module handles model training, evaluation, and prediction.
"""

import pandas as pd
import numpy as np
import logging
from typing import Tuple, Dict, Any, Optional
from pathlib import Path
import joblib
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score, roc_auc_score
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings('ignore')

from config import (
    FEATURES_FILE, ATTACK_CLASSIFIER_MODEL, THREAT_SCORER_MODEL,
    RF_PARAMS, XGB_PARAMS, TEST_SIZE, RANDOM_STATE,
    LOG_FORMAT, LOG_LEVEL
)

# Configure logging
logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)


class HoneypotModel:
    """
    Handles machine learning model training and evaluation for honeypot attack classification.
    """
    
    def __init__(self, data_path: Optional[Path] = None):
        """
        Initialize the HoneypotModel.
        
        Args:
            data_path: Path to the engineered features file.
        """
        self.data_path = data_path or FEATURES_FILE
        self.df = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.feature_columns = None
        self.attack_classifier = None
        self.threat_scorer = None
        self.models = {}
        self.evaluation_results = {}
        
    def load_data(self) -> pd.DataFrame:
        """
        Load engineered features data.
        
        Returns:
            DataFrame containing engineered features.
        """
        logger.info(f"Loading feature data from {self.data_path}")
        try:
            self.df = pd.read_csv(self.data_path)
            logger.info(f"Loaded {len(self.df)} records")
            return self.df
        except FileNotFoundError:
            logger.error(f"Feature data file not found: {self.data_path}")
            raise
    
    def prepare_data_for_training(self, target: str = 'attack_type') -> None:
        """
        Prepare data for model training.
        
        Args:
            target: Target variable name ('attack_type' or 'severity').
        """
        logger.info(f"Preparing data for training (target: {target})...")
        
        if self.df is None:
            self.load_data()
        
        # Define feature columns
        exclude_columns = [
            'time', 'payload', 'from', 'country', 'timestamp',
            'attack_type', 'severity', 'date', 'month'
        ]
        
        self.feature_columns = [
            col for col in self.df.columns 
            if col not in exclude_columns and self.df[col].dtype in ['int64', 'float64']
        ]
        
        # Remove any columns with NaN or inf values
        valid_columns = []
        for col in self.feature_columns:
            if not self.df[col].isnull().any() and not np.isinf(self.df[col]).any():
                valid_columns.append(col)
            else:
                logger.warning(f"Removing column {col} due to NaN or inf values")
        
        self.feature_columns = valid_columns
        
        # Prepare features and target
        X = self.df[self.feature_columns].copy()
        y = self.df[target].copy()
        
        # Handle any remaining issues
        X = X.fillna(0)
        X = X.replace([np.inf, -np.inf], 0)
        
        # Split data
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
        )
        
        logger.info(f"Training set: {len(self.X_train)} samples")
        logger.info(f"Test set: {len(self.X_test)} samples")
        logger.info(f"Features: {len(self.feature_columns)}")
        logger.info(f"Classes: {y.nunique()}")
    
    def train_random_forest(self, target: str = 'attack_type') -> RandomForestClassifier:
        """
        Train Random Forest classifier.
        
        Args:
            target: Target variable name.
            
        Returns:
            Trained Random Forest model.
        """
        logger.info(f"Training Random Forest classifier for {target}...")
        
        if self.X_train is None:
            self.prepare_data_for_training(target)
        
        # Initialize model
        rf_model = RandomForestClassifier(**RF_PARAMS)
        
        # Train model
        rf_model.fit(self.X_train, self.y_train)
        
        # Store model
        model_name = f'random_forest_{target}'
        self.models[model_name] = rf_model
        
        logger.info(f"Random Forest training complete for {target}")
        
        return rf_model
    
    def train_xgboost(self, target: str = 'attack_type') -> XGBClassifier:
        """
        Train XGBoost classifier.
        
        Args:
            target: Target column name ('attack_type' or 'severity').
            
        Returns:
            Trained XGBoost model.
        """
        logger.info(f"Training XGBoost classifier for {target}...")
        
        if self.X_train is None:
            self.prepare_data_for_training(target)
        
        # Encode labels for XGBoost (it requires integer labels)
        from sklearn.preprocessing import LabelEncoder
        le = LabelEncoder()
        y_train_encoded = le.fit_transform(self.y_train)
        y_test_encoded = le.transform(self.y_test)
        
        # Store original y_test for evaluation
        y_test_original = self.y_test.copy()
        self.y_test = y_test_encoded
        
        # Initialize model
        xgb_model = XGBClassifier(**XGB_PARAMS)
        
        # Train model
        xgb_model.fit(self.X_train, y_train_encoded)
        
        # Store the label encoder for later use
        xgb_model.label_encoder_ = le
        
        # Restore original y_test after storing encoded version
        self.y_test = y_test_original
        
        # Store model
        model_name = f'xgboost_{target}'
        self.models[model_name] = xgb_model
        
        logger.info(f"XGBoost training complete for {target}")
        
        return xgb_model
    
    def evaluate_model(self, model: Any, model_name: str) -> Dict[str, Any]:
        """
        Evaluate model performance.
        
        Args:
            model: Trained model to evaluate.
            model_name: Name of the model.
            
        Returns:
            Dictionary containing evaluation metrics.
        """
        logger.info(f"Evaluating model: {model_name}...")
        
        # Predictions
        y_pred = model.predict(self.X_test)
        
        # Decode predictions if model has label encoder (XGBoost)
        if hasattr(model, 'label_encoder_'):
            y_pred = model.label_encoder_.inverse_transform(y_pred)
        
        y_pred_proba = model.predict_proba(self.X_test) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(self.y_test, y_pred)
        
        # Handle multi-class metrics
        average_method = 'weighted'
        precision = precision_score(self.y_test, y_pred, average=average_method, zero_division=0)
        recall = recall_score(self.y_test, y_pred, average=average_method, zero_division=0)
        f1 = f1_score(self.y_test, y_pred, average=average_method, zero_division=0)
        
        # Classification report
        class_report = classification_report(self.y_test, y_pred, zero_division=0)
        
        # Confusion matrix
        conf_matrix = confusion_matrix(self.y_test, y_pred)
        
        # Cross-validation score (skipping for speed in demo)
        try:
            # Skip cross-validation to speed up training for demo
            logger.info("Skipping cross-validation for faster execution")
            cv_mean = None
            cv_std = None
        except Exception as e:
            logger.warning(f"Cross-validation failed: {str(e)}")
            cv_mean = None
            cv_std = None
        
        # Feature importance
        feature_importance = None
        if hasattr(model, 'feature_importances_'):
            feature_importance = pd.DataFrame({
                'feature': self.feature_columns,
                'importance': model.feature_importances_
            }).sort_values('importance', ascending=False)
        
        # Compile results
        results = {
            'model_name': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'classification_report': class_report,
            'confusion_matrix': conf_matrix,
            'cv_mean': cv_mean,
            'cv_std': cv_std,
            'feature_importance': feature_importance,
            'test_predictions': y_pred,
            'test_probabilities': y_pred_proba
        }
        
        # Store results
        self.evaluation_results[model_name] = results
        
        # Log metrics
        logger.info(f"{model_name} - Accuracy: {accuracy:.4f}")
        logger.info(f"{model_name} - Precision: {precision:.4f}")
        logger.info(f"{model_name} - Recall: {recall:.4f}")
        logger.info(f"{model_name} - F1 Score: {f1:.4f}")
        if cv_mean is not None:
            logger.info(f"{model_name} - CV Score: {cv_mean:.4f} (+/- {cv_std:.4f})")
        
        return results
    
    def train_attack_classifier(self) -> None:
        """Train and evaluate attack type classifier."""
        logger.info("Training attack type classifier...")
        
        # Prepare data
        self.prepare_data_for_training(target='attack_type')
        
        # Train Random Forest
        rf_model = self.train_random_forest(target='attack_type')
        self.attack_classifier = rf_model
        
        # Evaluate
        self.evaluate_model(rf_model, 'attack_classifier_rf')
        
        # Train XGBoost
        xgb_model = self.train_xgboost(target='attack_type')
        
        # Evaluate
        self.evaluate_model(xgb_model, 'attack_classifier_xgb')
        
        logger.info("Attack classifier training complete")
    
    def train_threat_scorer(self) -> None:
        """Train and evaluate threat severity scorer."""
        logger.info("Training threat severity scorer...")
        
        # Prepare data
        self.prepare_data_for_training(target='severity')
        
        # Train Random Forest
        rf_model = self.train_random_forest(target='severity')
        self.threat_scorer = rf_model
        
        # Evaluate
        self.evaluate_model(rf_model, 'threat_scorer_rf')
        
        # Train XGBoost
        xgb_model = self.train_xgboost(target='severity')
        
        # Evaluate
        self.evaluate_model(xgb_model, 'threat_scorer_xgb')
        
        logger.info("Threat scorer training complete")
    
    def save_models(self) -> None:
        """Save trained models to disk."""
        ATTACK_CLASSIFIER_MODEL.parent.mkdir(parents=True, exist_ok=True)
        
        if self.attack_classifier:
            joblib.dump(self.attack_classifier, ATTACK_CLASSIFIER_MODEL)
            logger.info(f"Attack classifier saved to {ATTACK_CLASSIFIER_MODEL}")
        
        if self.threat_scorer:
            joblib.dump(self.threat_scorer, THREAT_SCORER_MODEL)
            logger.info(f"Threat scorer saved to {THREAT_SCORER_MODEL}")
        
        # Save feature columns
        feature_file = ATTACK_CLASSIFIER_MODEL.parent / 'feature_columns.pkl'
        joblib.dump(self.feature_columns, feature_file)
        logger.info(f"Feature columns saved to {feature_file}")
    
    def load_models(self) -> None:
        """Load pre-trained models from disk."""
        try:
            self.attack_classifier = joblib.load(ATTACK_CLASSIFIER_MODEL)
            logger.info(f"Attack classifier loaded from {ATTACK_CLASSIFIER_MODEL}")
        except FileNotFoundError:
            logger.warning("Attack classifier model not found")
        
        try:
            self.threat_scorer = joblib.load(THREAT_SCORER_MODEL)
            logger.info(f"Threat scorer loaded from {THREAT_SCORER_MODEL}")
        except FileNotFoundError:
            logger.warning("Threat scorer model not found")
        
        try:
            feature_file = ATTACK_CLASSIFIER_MODEL.parent / 'feature_columns.pkl'
            self.feature_columns = joblib.load(feature_file)
            logger.info(f"Feature columns loaded from {feature_file}")
        except FileNotFoundError:
            logger.warning("Feature columns file not found")
    
    def predict_attack_type(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict attack type for new data.
        
        Args:
            features: DataFrame containing feature values.
            
        Returns:
            Array of predicted attack types.
        """
        if self.attack_classifier is None:
            raise ValueError("Attack classifier not trained. Train or load model first.")
        
        return self.attack_classifier.predict(features)
    
    def predict_threat_severity(self, features: pd.DataFrame) -> np.ndarray:
        """
        Predict threat severity for new data.
        
        Args:
            features: DataFrame containing feature values.
            
        Returns:
            Array of predicted threat severity levels.
        """
        if self.threat_scorer is None:
            raise ValueError("Threat scorer not trained. Train or load model first.")
        
        return self.threat_scorer.predict(features)
    
    def get_evaluation_summary(self) -> pd.DataFrame:
        """
        Get summary of model evaluation results.
        
        Returns:
            DataFrame containing evaluation metrics for all models.
        """
        summary_data = []
        
        for model_name, results in self.evaluation_results.items():
            summary_data.append({
                'Model': model_name,
                'Accuracy': results['accuracy'],
                'Precision': results['precision'],
                'Recall': results['recall'],
                'F1 Score': results['f1_score'],
                'CV Mean': results.get('cv_mean', None),
                'CV Std': results.get('cv_std', None)
            })
        
        return pd.DataFrame(summary_data)


def main():
    """Main function to run model training pipeline."""
    model = HoneypotModel()
    
    # Train attack classifier
    model.train_attack_classifier()
    
    # Train threat scorer
    model.train_threat_scorer()
    
    # Save models
    model.save_models()
    
    # Print evaluation summary
    print("\n" + "="*80)
    print("MODEL EVALUATION SUMMARY")
    print("="*80)
    summary = model.get_evaluation_summary()
    print(summary.to_string(index=False))
    print("="*80)
    
    # Print detailed results
    for model_name, results in model.evaluation_results.items():
        print(f"\n\n{model_name.upper()} - Detailed Results:")
        print("-" * 80)
        print(results['classification_report'])
        
        if results['feature_importance'] is not None:
            print("\nTop 10 Feature Importances:")
            print(results['feature_importance'].head(10).to_string(index=False))


if __name__ == "__main__":
    main()
