import sys
sys.path.append('./src')
from data_parser import DataParser
from feature_engineering import FeatureEngineer

# Load data
parser = DataParser('data/raw/AI_RESUME_SCREENING_AUGMENTED.csv')
df = parser.load_data()
df = parser.clean_data()
df = parser.encode_target()
X, y = parser.prepare_for_modeling()

# Engineer features
engineer = FeatureEngineer()
X_engineered = engineer.create_all_features(X)
X_final = engineer.get_model_features(X_engineered)

print(f"Total features: {X_final.shape[1]}")
print(f"Feature dtypes:\n{X_final.dtypes}")

# Check numerical columns
numerical_cols = X_final.select_dtypes(include=['number']).columns.tolist()
print(f"\nNumerical columns: {len(numerical_cols)}")
print(f"Numerical cols: {sorted(numerical_cols)}")

# Check non-numerical
non_numerical = [c for c in X_final.columns if c not in numerical_cols]
if non_numerical:
    print(f"\n⚠️ NON-NUMERICAL COLUMNS FOUND: {non_numerical}")
    print(f"Sample values:")
    for col in non_numerical:
        print(f"  {col}: {X_final[col].head().tolist()}")
