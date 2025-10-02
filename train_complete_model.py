"""
Complete Model Training Pipeline

"""

import sys
sys.path.append('./src')
import os

model_dir = '../models'
os.makedirs(model_dir, exist_ok=True)

from data_parser import DataParser
from feature_engineering import FeatureEngineer
from model import JobFitnessModel, train_and_compare_models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

def main():
    print("="*70)
    print("JOB FITNESS SCORING SYSTEM - COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load augmented data
    print("\n[1/6] Loading augmented dataset...")
    parser = DataParser('data/raw/AI_RESUME_SCREENING_AUGMENTED.csv')
    df = parser.load_data()
    df = parser.clean_data()
    df = parser.encode_target()
    
    # Step 2: Prepare features
    print("\n[2/6] Preparing features...")
    X, y = parser.prepare_for_modeling()
    print(f"Features: {X.shape}")
    print(f"Target distribution: {y.value_counts().to_dict()}")
    
    # Step 3: Feature engineering
    print("\n[3/6] Engineering features...")
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    X_final = engineer.get_model_features(X_engineered)
    print(f"Engineered features: {X_final.shape}")
    
    # Step 4: Train-test split
    print("\n[4/6] Splitting data...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_final, y, 
        test_size=0.2, 
        random_state=42, 
        stratify=y
    )
    
    # Scale features
    X_train_scaled = engineer.scale_features(X_train, fit=True)
    X_test_scaled = engineer.scale_features(X_test, fit=False)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Step 5: Train models
    print("\n[5/6] Training models...")
    models = train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 6: Save best model
    print("\n[6/6] Saving models...")
    
    # Save XGBoost model
    xgb_model = models['xgboost']
    xgb_model.save_model('models/xgboost_job_fitness.pkl')
    xgb_model.plot_confusion_matrix(X_test_scaled, y_test)
    xgb_model.plot_feature_importance()
    
    # Save Random Forest model
    rf_model = models['random_forest']
    rf_model.save_model('models/rf_job_fitness.pkl')
    
    engineer.save('models/feature_engineer.pkl')

    # Cross-validation on best model
    print("\n" + "="*70)
    print("CROSS-VALIDATION ON BEST MODEL")
    print("="*70)
    xgb_model.cross_validate(X_train_scaled, y_train, cv=5)
    
    print("\n" + "="*70)
    print("TRAINING COMPLETE!")
    print("="*70)
    print("Models saved to:")
    print("  - models/xgboost_job_fitness.pkl")
    print("  - models/rf_job_fitness.pkl")
    print("\nVisualization saved to:")
    print("  - models/confusion_matrix_xgboost.png")
    print("  - models/feature_importance_xgboost.png")


if __name__ == "__main__":
    main()