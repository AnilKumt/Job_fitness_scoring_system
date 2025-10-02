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
import os

class JobFitnessModel:
    def __init__(self, model_type='xgboost'):
        self.model_type = model_type
        self.model = None
        self.feature_names = None
        self.metrics = {}

        if model_type == 'xgboost':
            self.model = xgb.XGBClassifier(
                n_estimators=100, max_depth=6, learning_rate=0.1,
                subsample=0.8, colsample_bytree=0.8, random_state=42,
                eval_metric='logloss'
            )
        elif model_type == 'random_forest':
            self.model = RandomForestClassifier(
                n_estimators=100, max_depth=10, random_state=42
            )
        else:
            raise ValueError("model_type must be 'xgboost' or 'random_forest'")

    def train(self, X_train: pd.DataFrame, y_train: pd.Series,
              X_val: pd.DataFrame = None, y_val: pd.Series = None) -> None:
        print(f"Training {self.model_type} model...")
        self.feature_names = X_train.columns.tolist()
        if self.model_type == 'xgboost' and X_val is not None:
            self.model.fit(
                X_train, y_train, eval_set=[(X_val, y_val)], verbose=False
            )
        else:
            self.model.fit(X_train, y_train)
        print("Training complete!")

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict(X)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        return self.model.predict_proba(X)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict:
        y_pred = self.predict(X_test)
        self.metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred)
        }
        print("\n" + "="*50 + "\nMODEL EVALUATION RESULTS\n" + "="*50)
        for key, value in self.metrics.items():
            print(f"{key.capitalize():<10}: {value:.4f}")
        print("\nClassification Report:\n", classification_report(y_test, y_pred, target_names=['Reject', 'Hire']))
        print("\nConfusion Matrix:\n", confusion_matrix(y_test, y_pred))
        return self.metrics

    def plot_confusion_matrix(self, X_test: pd.DataFrame, y_test: pd.Series) -> None:
        os.makedirs('models', exist_ok=True)
        y_pred = self.predict(X_test)
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8, 6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Reject', 'Hire'], yticklabels=['Reject', 'Hire'])
        plt.title(f'Confusion Matrix - {self.model_type.upper()}')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        path = f'models/confusion_matrix_{self.model_type}.png'
        plt.savefig(path)
        print(f"Confusion matrix saved to {path}")

    def plot_feature_importance(self, top_n: int = 15) -> None:
        os.makedirs('models', exist_ok=True)
        importance = self.model.feature_importances_
        feature_importance = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False).head(top_n)
        plt.figure(figsize=(10, 8))
        sns.barplot(data=feature_importance, y='feature', x='importance')
        plt.title(f'Top {top_n} Feature Importances - {self.model_type.upper()}')
        plt.xlabel('Importance Score')
        plt.tight_layout()
        path = f'models/feature_importance_{self.model_type}.png'
        plt.savefig(path)
        print(f"Feature importance plot saved to {path}")

    # --- THIS IS THE MISSING METHOD ---
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, cv: int = 5) -> dict:
        """
        Perform cross-validation
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
            print(f"{metric.capitalize():<10}: {scores.mean():.4f} (+/- {scores.std() * 2:.4f})")
        return cv_scores
    # ------------------------------------

    def save_model(self, filepath: str = 'models/job_fitness_model.pkl') -> None:
        """Save the trained model and feature names"""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        data_to_save = {'model': self.model, 'feature_names': self.feature_names}
        joblib.dump(data_to_save, filepath)
        print(f"\nModel and feature names saved to {filepath}")

    def load_model(self, filepath: str = 'models/job_fitness_model.pkl') -> None:
        """Load a trained model and feature names"""
        data_loaded = joblib.load(filepath)
        self.model = data_loaded['model']
        self.feature_names = data_loaded['feature_names']
        print(f"Model and feature names loaded from {filepath}")

# ... (the rest of the file is the same)
def train_and_compare_models(X_train, X_test, y_train, y_test):
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