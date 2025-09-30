"""
Model Training Module
Trains and evaluates XGBoost classifier for job fitness scoring
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, classification_report
)
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier
import joblib
import matplotlib.pyplot as plt
import seaborn as sns


class JobFitnessModel:
    def __init__(self, model_type='xgboost'):
        """
        Initialize the Job Fitness Scoring Model
        
        Args:
            model_type: 'xgboost' or 'random_forest'
        """
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.metrics = {}
        
        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100,
                max_depth=10,
                random_state=42
            )
        else:
            raise ValueError("model_type must be 'xgboost' or 'random_forest'")
    
    def train(self, X_train: pd.DataFrame, y_train: pd.Series, 
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        """
        Train the model
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        print(f"Training {self.model_type} model...")
        self.feature_names = X_train.columns.tolist()
        
        if self.model_type == 'xgboost' and X_val is not None:
            # Use early stopping with validation set
            self.model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        
        print("Training complete!")
    
    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Make predictions"""
        return self.model.predict(X)
    
    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        """Get prediction probabilities"""
        return self.model.predict_proba(X)
    
    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        """
        Evaluate the model and return metrics
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_pred_proba = self.predict_proba(X_test)[:, 1]
        
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        
        print("\n" + "="*50)
        print("MODEL EVALUATION RESULTS")
        print("="*50)
        print(f"Accuracy:  {self.metrics['accuracy']:.4f}")
        print(f"Precision: {self.metrics['precision']:.4f}")
        print(f"Recall:    {self.metrics['recall']:.4f}")
        print(f"F1-Score:  {self.metrics['f1_score']:.4f}")
        
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred, 
                                   target_names=['Reject', 'Hire']))
        
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred)
        print(cm)
        
        return self.metrics
    
    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        """Plot confusion matrix"""
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Reject', 'Hire'],
                   yticklabels=['Reject', 'Hire'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(f'../models/confusion_matrix_{self.model_type}.png')
        print(f"Confusion matrix saved to models/confusion_matrix_{self.model_type}.png")
    
    def plot_feature_importance(self, top_n: int = 15) -> None:
        """
        Plot feature importance
        
        Args:
            top_n: Number of top features to display
        """
        if self.model_type == 'xgboost':
            importance = self.model.feature_importances_
        elif self.model_type == 'random_forest':
            importance = self.model.feature_importances_
        else:
            print("Feature importance not available for this model type")
            return
        
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type.upper()}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        plt.savefig(f'../models/feature_importance_{self.model_type}.png')
        print(f"Feature importance plot saved to models/feature_importance_{self.model_type}.png")
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """
        Perform cross-validation
        
        Args:
            X: Features
            y: Target
            cv: Number of folds
            
        Returns:
            Dictionary with cross-validation scores
        """
        print(f"\nPerforming {cv}-fold cross-validation...")
        
        cv_scores = {
            'accuracy': cross_val_score(self.model, X, y, cv=cv, scoring='accuracy'),
            'precision': cross_val_score(self.model, X, y, cv=cv, scoring='precision'),
            'recall': cross_val_score(self.model, X, y, cv=cv, scoring='recall'),
            'f1': cross_val_score(self.model, X, y, cv=cv, scoring='f1')
        }
        
        print("\nCross-Validation Results:")
        for metric, scores in cv_scores.items():
            print(f"{metric.capitalize():10s}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        
        return cv_scores
    
    def save_model(self, filepath: str = '../models/job_fitness_model.pkl') -> None:
        """Save the trained model"""
        joblib.dump(self.model, filepath)
        print(f"\nModel saved to {filepath}")
    
    def load_model(self, filepath: str = '../models/job_fitness_model.pkl') -> None:
        """Load a trained model"""
        self.model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")


def train_and_compare_models(X_train, X_test, y_train, y_test):
    """
    Train both XGBoost and Random Forest models and compare
    
    Args:
        X_train, X_test, y_train, y_test: Train/test splits
        
    Returns:
        Dictionary with both trained models
    """
    models = {}
    
    # Train XGBoost
    print("\n" + "="*60)
    print("TRAINING XGBOOST MODEL")
    print("="*60)
    xgb_model = JobFitnessModel(model_type='xgboost')
    xgb_model.train(X_train, y_train, X_test, y_test)
    xgb_metrics = xgb_model.evaluate(X_test, y_test)
    models['xgboost'] = xgb_model
    
    # Train Random Forest
    print("\n" + "="*60)
    print("TRAINING RANDOM FOREST MODEL")
    print("="*60)
    rf_model = JobFitnessModel(model_type='random_forest')
    rf_model.train(X_train, y_train)
    rf_metrics = rf_model.evaluate(X_test, y_test)
    models['random_forest'] = rf_model
    
    # Compare models
    print("\n" + "="*60)
    print("MODEL COMPARISON")
    print("="*60)
    comparison = pd.DataFrame({
        'XGBoost': xgb_metrics,
        'Random Forest': rf_metrics
    })
    print(comparison)
    
    # Determine best model
    best_model_name = 'XGBoost' if xgb_metrics['f1_score'] > rf_metrics['f1_score'] else 'Random Forest'
    print(f"\nBest Model (by F1-Score): {best_model_name}")
    
    return models


if __name__ == "__main__":
    from data_parser import DataParser
    from feature_engineering import FeatureEngineer
    
    # Load and prepare data
    parser = DataParser('../data/raw/your_dataset.csv')
    df = parser.load_data()
    df = parser.clean_data()
    X, y = parser.prepare_for_modeling()
    
    # Engineer features
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    X_final = engineer.get_model_features(X_engineered)
    X_scaled = engineer.scale_features(X_final)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X_scaled, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"\nTrain set: {X_train.shape}")
    print(f"Test set: {X_test.shape}")
    
    # Train and compare models
    models = train_and_compare_models(X_train, X_test, y_train, y_test)
    
    # Save best model
    best_model = models['xgboost']  # You can change this based on comparison
    best_model.save_model()
    best_model.plot_confusion_matrix(X_test, y_test)
    best_model.plot_feature_importance()
