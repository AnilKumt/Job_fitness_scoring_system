"""
Feature Engineering Module
Extracts and engineers features from resume data for ML model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from typing import List, Dict, Tuple
import re


class FeatureEngineer:
    def __init__(self):
        """Initialize the Feature Engineer"""
        self.education_encoder = LabelEncoder()
        self.role_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        self.skill_list = None
        
    def extract_skill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract skill-based features from the Skills column
        
        Args:
            df: DataFrame with 'Skills' column
            
        Returns:
            DataFrame with skill features
        """
        df = df.copy()
        
        # Count number of skills
        df['skill_count'] = df['Skills'].apply(
            lambda x: len([s.strip() for s in str(x).split(',') if s.strip()])
        )
        
        # Extract specific skill categories
        df['has_python'] = df['Skills'].str.contains('Python', case=False, na=False).astype(int)
        df['has_ml'] = df['Skills'].str.contains('Machine Learning|ML|Deep Learning', case=False, na=False).astype(int)
        df['has_sql'] = df['Skills'].str.contains('SQL|Database', case=False, na=False).astype(int)
        df['has_cloud'] = df['Skills'].str.contains('AWS|Azure|GCP|Cloud', case=False, na=False).astype(int)
        df['has_data_viz'] = df['Skills'].str.contains('Tableau|Power BI|Visualization', case=False, na=False).astype(int)
        df['has_nlp'] = df['Skills'].str.contains('NLP|Natural Language', case=False, na=False).astype(int)
        df['has_cv'] = df['Skills'].str.contains('Computer Vision|CV|OpenCV', case=False, na=False).astype(int)
        
        return df
    
    def extract_certification_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Extract features from Certifications column
        
        Args:
            df: DataFrame with 'Certifications' column
            
        Returns:
            DataFrame with certification features
        """
        df = df.copy()
        
        # Has any certification
        df['has_certification'] = (df['Certifications'] != 'None').astype(int)
        
        # Count certifications
        df['cert_count'] = df['Certifications'].apply(
            lambda x: 0 if x == 'None' else len([c.strip() for c in str(x).split(',') if c.strip()])
        )
        
        # Specific certification types
        df['has_google_cert'] = df['Certifications'].str.contains('Google', case=False, na=False).astype(int)
        df['has_aws_cert'] = df['Certifications'].str.contains('AWS', case=False, na=False).astype(int)
        df['has_dl_cert'] = df['Certifications'].str.contains('Deep Learning', case=False, na=False).astype(int)
        
        return df
    
    def encode_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Encode education level as ordinal variable
        
        Args:
            df: DataFrame with 'Education' column
            
        Returns:
            DataFrame with encoded education
        """
        df = df.copy()
        
        # Define education hierarchy
        education_map = {
            'B.Sc': 1,
            'B.Tech': 1,
            'Bachelor': 1,
            'MBA': 2,
            'M.Sc': 2,
            'Master': 2,
            'PhD': 3,
            'Ph.D': 3
        }
        
        # Map education to ordinal values
        df['education_level'] = df['Education'].map(education_map)
        
        # Fill any unmapped values with 1 (Bachelor equivalent)
        df['education_level'].fillna(1, inplace=True)
        
        return df
    
    def encode_job_role(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode job roles
        
        Args:
            df: DataFrame with 'Job Role' column
            
        Returns:
            DataFrame with one-hot encoded job roles
        """
        df = df.copy()
        
        # One-hot encode job roles
        role_dummies = pd.get_dummies(df['Job Role'], prefix='role')
        df = pd.concat([df, role_dummies], axis=1)
        
        return df
    
    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Create interaction features between different variables
        
        Args:
            df: DataFrame with base features
            
        Returns:
            DataFrame with interaction features
        """
        df = df.copy()
        
        # Experience * Projects
        df['exp_projects_interaction'] = df['Experience (Years)'] * df['Projects Count']
        
        # Experience * Education
        if 'education_level' in df.columns:
            df['exp_education_interaction'] = df['Experience (Years)'] * df['education_level']
        
        # Skills * Projects
        if 'skill_count' in df.columns:
            df['skills_projects_interaction'] = df['skill_count'] * df['Projects Count']
        
        # Experience bins (junior, mid, senior)
        df['experience_level'] = pd.cut(
            df['Experience (Years)'],
            bins=[0, 2, 5, 100],
            labels=['junior', 'mid', 'senior']
        )
        
        return df
    
    def create_all_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Create all engineered features
        
        Args:
            X: Raw feature DataFrame
            
        Returns:
            Fully engineered feature DataFrame
        """
        print("Starting feature engineering...")
        
        # Extract skill features
        X = self.extract_skill_features(X)
        print(f"✓ Extracted skill features")
        
        # Extract certification features
        X = self.extract_certification_features(X)
        print(f"✓ Extracted certification features")
        
        # Encode education
        X = self.encode_education(X)
        print(f"✓ Encoded education levels")
        
        # Encode job roles
        X = self.encode_job_role(X)
        print(f"✓ Encoded job roles")
        
        # Create interaction features
        X = self.create_interaction_features(X)
        print(f"✓ Created interaction features")
        
        print(f"\nFinal feature count: {X.shape[1]}")
        
        return X
    
    def get_model_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Select and prepare final features for modeling
        
        Args:
            X: Engineered feature DataFrame
            
        Returns:
            Clean feature DataFrame ready for modeling
        """
        # Drop original text columns and categorical variables
        drop_cols = ['Skills', 'Certifications', 'Job Role', 'Education', 'experience_level']
        
        X_model = X.drop(columns=[col for col in drop_cols if col in X.columns])
        
        print(f"\nFinal model features ({X_model.shape[1]}):")
        print(X_model.columns.tolist())
        
        return X_model
    
    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """
        Scale numerical features
        
        Args:
            X: Feature DataFrame
            fit: Whether to fit the scaler (True for training, False for test)
            
        Returns:
            Scaled feature DataFrame
        """
        numerical_cols = X.select_dtypes(include=[np.number]).columns
        
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            X[numerical_cols] = self.scaler.transform(X[numerical_cols])
        
        return X


if __name__ == "__main__":
    # Example usage
    from data_parser import DataParser
    
    # Load data
    parser = DataParser('../data/raw/your_dataset.csv')
    df = parser.load_data()
    df = parser.clean_data()
    X, y = parser.prepare_for_modeling()
    
    # Engineer features
    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    X_final = engineer.get_model_features(X_engineered)
    X_scaled = engineer.scale_features(X_final)
    
    print(f"\nReady for modeling!")
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
