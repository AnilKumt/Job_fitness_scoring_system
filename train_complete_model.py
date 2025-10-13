"""
Complete Model Training Pipeline - FIXED VERSION
"""

import sys
sys.path.append('./src')
import os

# Create models directory in the correct location
model_dir = 'models'
os.makedirs(model_dir, exist_ok=True)

from data_parser import DataParser
from feature_engineering import FeatureEngineer
from model import JobFitnessModel, train_and_compare_models
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import joblib

def main():
    print("="*70)
    print("JOB FITNESS SCORING SYSTEM - COMPLETE TRAINING PIPELINE")
    print("="*70)
    
    # Step 1: Load augmented data
    print("\n[1/6] Loading augmented dataset...")
    try:
        parser = DataParser('data/raw/AI_RESUME_SCREENING_AUGMENTED.csv')
        df = parser.load_data()
        df = parser.clean_data()
        df = parser.encode_target()
    except FileNotFoundError:
        print("ERROR: Augmented dataset not found!")
        print("Please run data_augmentation.py first to create the dataset.")
        return
    
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
    X_train_scaled = engineer.scale_features(X_train.copy(), fit=True)
    X_test_scaled = engineer.scale_features(X_test.copy(), fit=False)
    
    print(f"Training set: {X_train_scaled.shape}")
    print(f"Test set: {X_test_scaled.shape}")
    
    # Step 5: Train models
    print("\n[5/6] Training models...")
    models = train_and_compare_models(X_train_scaled, X_test_scaled, y_train, y_test)
    
    # Step 6: Save best model and feature engineer
    print("\n[6/6] Saving models and preprocessors...")
    
    # Save XGBoost model
    xgb_model = models['xgboost']
    xgb_model.save_model('models/xgboost_job_fitness.pkl')
    
    # Save Random Forest model
    rf_model = models['random_forest']
    rf_model.save_model('models/rf_job_fitness.pkl')
    
    # IMPORTANT: Save the feature engineer (with fitted scaler)
    joblib.dump(engineer, 'models/feature_engineer.pkl')
    print("Feature engineer saved to models/feature_engineer.pkl")
    
    # Save visualizations
    try:
        xgb_model.plot_confusion_matrix(X_test_scaled, y_test)
        xgb_model.plot_feature_importance()
    except Exception as e:
        print(f"Warning: Could not save visualizations: {e}")
    
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
    print("  - models/feature_engineer.pkl (IMPORTANT)")
    print("\nVisualization saved to:")
    print("  - models/confusion_matrix_xgboost.png")
    print("  - models/feature_importance_xgboost.png")
    
    # Test prediction
    print("\n" + "="*70)
    print("TESTING PREDICTION")
    print("="*70)
    sample = X_test_scaled.iloc[0:1]
    prediction = xgb_model.predict(sample)
    probability = xgb_model.predict_proba(sample)
    print(f"Sample prediction: {'HIRE' if prediction[0] == 1 else 'REJECT'}")
    print(f"Hire probability: {probability[0][1]:.2%}")


if __name__ == "__main__":
    main()
