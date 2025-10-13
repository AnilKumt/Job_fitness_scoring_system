"""
Data Parser Module - FINAL VERSION
Loads and preprocesses the resume dataset
"""

import pandas as pd
import numpy as np
from typing import Tuple

class DataParser:
    def __init__(self, filepath: str):
        """
        Initialize the DataParser with dataset filepath
        
        Args:
            filepath: Path to the CSV file
        """
        self.filepath = filepath
        self.df = None

    def load_data(self) -> pd.DataFrame:
        """Load the dataset from CSV"""
        self.df = pd.read_csv(self.filepath)
        print(f"Dataset loaded: {self.df.shape[0]} rows, {self.df.shape[1]} columns")
        return self.df

    def clean_data(self) -> pd.DataFrame:
        """Clean the dataset by handling missing values and duplicates"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        original_shape = self.df.shape

        # Create explicit copy to avoid warnings
        self.df = self.df.copy()
        self.df['Certifications'] = self.df['Certifications'].fillna('None')

        # Remove duplicates
        if 'Resume_ID' in self.df.columns:
            self.df.drop_duplicates(subset=['Resume_ID'], inplace=True)
        else:
            self.df.drop_duplicates(inplace=True)

        # Remove rows with missing critical information
        critical_cols = ['Skills', 'Experience (Years)', 'Education', 'Job Role', 'Recruiter Decision']
        self.df.dropna(subset=critical_cols, inplace=True)

        print(f"Data cleaned: {original_shape} -> {self.df.shape}")
        print(f"Removed {original_shape[0] - self.df.shape[0]} rows")

        return self.df

    def encode_target(self) -> pd.DataFrame:
        """
        Encode the target variable 'Recruiter Decision' as binary
        Hire = 1, Reject = 0
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        self.df['Target'] = self.df['Recruiter Decision'].apply(
            lambda x: 1 if x == 'Hire' else 0
        )

        print(f"\nTarget Distribution:")
        print(self.df['Target'].value_counts())
        print(f"Hire Rate: {self.df['Target'].mean():.2%}")

        return self.df

    def get_data_info(self) -> None:
        """Print detailed information about the dataset"""
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        print("\n" + "="*50)
        print("DATASET INFORMATION")
        print("="*50)

        print(f"\nShape: {self.df.shape}")
        print(f"\nColumns: {self.df.columns.tolist()}")

        print("\nMissing Values:")
        missing = self.df.isnull().sum()
        print(missing[missing > 0] if missing.sum() > 0 else "No missing values")

        print("\nData Types:")
        print(self.df.dtypes)

        print("\nNumerical Statistics:")
        print(self.df.describe())

        print("\nJob Roles:")
        print(self.df['Job Role'].value_counts())

        print("\nEducation Levels:")
        print(self.df['Education'].value_counts())

    def prepare_for_modeling(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare data for model training
        
        Returns:
            Tuple of (features_df, target_series)
        """
        if self.df is None:
            raise ValueError("Data not loaded. Call load_data() first.")

        if 'Target' not in self.df.columns:
            self.encode_target()

        feature_cols = [
            'Skills', 'Experience (Years)', 'Education', 
            'Certifications', 'Job Role', 'Projects Count', 'AI Score (0-100)'
        ]

        X = self.df[feature_cols].copy()
        y = self.df['Target'].copy()

        return X, y


if __name__ == "__main__":
    parser = DataParser('data/raw/AI_Resume_Screening.csv')
    df = parser.load_data()
    df = parser.clean_data()
    df = parser.encode_target()
    parser.get_data_info()
    X, y = parser.prepare_for_modeling()
    print(f"\nFeatures shape: {X.shape}")
    print(f"Target shape: {y.shape}")
