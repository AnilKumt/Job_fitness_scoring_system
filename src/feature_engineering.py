"""
Feature Engineering Module - FINAL VERSION
Extracts and engineers features from resume data for ML model
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

class FeatureEngineer:
    def __init__(self):
        """Initialize the Feature Engineer"""
        self.scaler = StandardScaler()

    def extract_skill_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract skill-based features from the Skills column"""
        df = df.copy()
        
        df['skill_count'] = df['Skills'].apply(
            lambda x: len([s.strip() for s in str(x).split(',') if s.strip()])
        )
        
        df['has_python'] = df['Skills'].str.contains('Python', case=False, na=False).astype(int)
        df['has_ml'] = df['Skills'].str.contains('Machine Learning|ML|Deep Learning', case=False, na=False).astype(int)
        df['has_sql'] = df['Skills'].str.contains('SQL|Database', case=False, na=False).astype(int)
        df['has_cloud'] = df['Skills'].str.contains('AWS|Azure|GCP|Cloud', case=False, na=False).astype(int)
        df['has_data_viz'] = df['Skills'].str.contains('Tableau|Power BI|Visualization', case=False, na=False).astype(int)
        df['has_nlp'] = df['Skills'].str.contains('NLP|Natural Language', case=False, na=False).astype(int)
        df['has_cv'] = df['Skills'].str.contains('Computer Vision|CV|OpenCV', case=False, na=False).astype(int)

        return df

    def extract_certification_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Extract features from Certifications column"""
        df = df.copy()
        
        df['has_certification'] = (df['Certifications'] != 'None').astype(int)
        
        df['cert_count'] = df['Certifications'].apply(
            lambda x: 0 if x == 'None' else len([c.strip() for c in str(x).split(',') if c.strip()])
        )
        
        df['has_google_cert'] = df['Certifications'].str.contains('Google', case=False, na=False).astype(int)
        df['has_aws_cert'] = df['Certifications'].str.contains('AWS', case=False, na=False).astype(int)
        df['has_dl_cert'] = df['Certifications'].str.contains('Deep Learning', case=False, na=False).astype(int)

        return df

    def encode_education(self, df: pd.DataFrame) -> pd.DataFrame:
        """Encode education level as ordinal variable"""
        df = df.copy()
        
        education_map = {
            'B.Sc': 1, 'B.Tech': 1, 'Bachelor': 1, 'b.s.': 1,
            'MBA': 2, 'M.Sc': 2, 'M.Tech': 2, 'Master': 2, "Master's": 2, 'master\'s': 2,
            'PhD': 3, 'Ph.D': 3, 'phd': 3
        }
        
        df['education_level'] = df['Education'].map(education_map)
        df['education_level'] = df['education_level'].fillna(1)

        return df

    def encode_job_role(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        One-hot encode job roles, ALWAYS including all possible roles.
        CRITICAL FIX: dtype=int ensures columns are integers, not booleans
        """
        df = df.copy()
        
        # Define all possible job roles
        ALL_JOB_ROLES = ['AI Researcher', 'Cybersecurity Analyst', 'Data Scientist', 'Software Engineer']
        
        # Create one-hot encoding with dtype=int (CRITICAL!)
        job_role_dummies = pd.get_dummies(df['Job Role'], prefix='role', dtype=int)
        
        # Ensure ALL possible role columns exist
        for role in ALL_JOB_ROLES:
            col = f'role_{role}'
            if col not in job_role_dummies.columns:
                job_role_dummies[col] = 0
        
        # Ensure consistent column order (alphabetical)
        role_columns = sorted([f'role_{role}' for role in ALL_JOB_ROLES])
        job_role_dummies = job_role_dummies[role_columns]
        
        df = pd.concat([df, job_role_dummies], axis=1)
        
        return df

    def create_interaction_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create interaction features between variables"""
        df = df.copy()

        df['exp_projects_interaction'] = df['Experience (Years)'] * df['Projects Count']
        
        if 'education_level' in df.columns:
            df['exp_education_interaction'] = df['Experience (Years)'] * df['education_level']
        
        if 'skill_count' in df.columns:
            df['skills_projects_interaction'] = df['skill_count'] * df['Projects Count']
        
        df['experience_level'] = pd.cut(
            df['Experience (Years)'],
            bins=[0, 2, 5, 100],
            labels=['junior', 'mid', 'senior']
        )

        return df

    def create_all_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Create all engineered features"""
        print("Starting feature engineering...")

        X = self.extract_skill_features(X)
        print(f"✓ Extracted skill features")

        X = self.extract_certification_features(X)
        print(f"✓ Extracted certification features")

        X = self.encode_education(X)
        print(f"✓ Encoded education levels")

        X = self.encode_job_role(X)
        print(f"✓ Encoded job roles")

        X = self.create_interaction_features(X)
        print(f"✓ Created interaction features")

        print(f"\nFinal feature count: {X.shape[1]}")

        return X

    def get_model_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """Select and prepare final features for modeling"""
        drop_cols = ['Skills', 'Certifications', 'Job Role', 'Education', 'experience_level']
        X_model = X.drop(columns=[col for col in drop_cols if col in X.columns])

        print(f"\nFinal model features ({X_model.shape[1]}):")
        print(X_model.columns.tolist())

        return X_model

    def scale_features(self, X: pd.DataFrame, fit: bool = True) -> pd.DataFrame:
        """Scale numerical features"""
        X = X.copy()
        numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
        
        if fit:
            X[numerical_cols] = self.scaler.fit_transform(X[numerical_cols])
        else:
            # Use .values to bypass sklearn's feature name validation
            X_scaled = self.scaler.transform(X[numerical_cols].values)
            X[numerical_cols] = X_scaled

        return X


if __name__ == "__main__":
    from data_parser import DataParser

    parser = DataParser('../data/raw/AI_RESUME_SCREENING_AUGMENTED.csv')
    df = parser.load_data()
    df = parser.clean_data()
    X, y = parser.prepare_for_modeling()

    engineer = FeatureEngineer()
    X_engineered = engineer.create_all_features(X)
    X_final = engineer.get_model_features(X_engineered)
    X_scaled = engineer.scale_features(X_final)

    print(f"\nReady for modeling!")
    print(f"Features shape: {X_scaled.shape}")
    print(f"Target shape: {y.shape}")
