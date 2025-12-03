import os
import kagglehub
import pandas as pd

os.environ['KAGGLEHUB_CACHE'] = '/Users/juliana_miller/Desktop/ece143-assignments/project/project_datasets'

path = kagglehub.dataset_download("uciml/breast-cancer-wisconsin-data")

print("Path to dataset files:", path)

df = pd.read_csv('project_datasets/breast_cancer_dia.csv')

# Take a look at the dataset
print("Original shape:", df.shape)
print("\n" + "="*80 + "\n")
print("Dataset info (column names and types):")
df.info()
print("\n" + "="*80 + "\n")
print("Dataset Description:", df.describe())
print("\n" + "="*80 + "\n")
print("Dataset First 5 rows:")
print(df.head())
print("\n" + "="*80 + "\n")

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())
print("\n" + "="*80 + "\n")

# Description of dataset showed 'Unnamed: 32' column
# Dropped 'Unnamed: 32' column
if 'Unnamed: 32' in df.columns:
    df = df.drop('Unnamed: 32', axis=1)

# Removed 'id' column as didn't serve a purpose for us
# Removed duplicates
df = df.drop('id', axis=1)
duplicates = df.duplicated().sum()
print(f"\nDuplicates found: {duplicates}")
df = df.drop_duplicates()
print("\n" + "="*80 + "\n")

# Changed 'diagnosis' column values from M/B to 1=malignant 0=benign to match other dataset
df['diagnosis'] = (df['diagnosis'] == 'M').astype(int)
print(f"Malignant: {df['diagnosis'].sum()}")
print(f"Benign: {(df['diagnosis'] == 0).sum()}")
print("\n" + "="*80 + "\n")

# Get all feature columns except for diagnosis (different scale)
feature_cols = [col for col in df.columns if col != 'diagnosis']

print("All features summary:")
print(f"Total features: {len(feature_cols)}")
print(f"Feature names: {feature_cols}")
print("\n" + "="*80 + "\n")

print("Cleaned dataset:")
print("Cleaned shape:", df.shape)
print("Cleaned Dataset info (column names and types):")
df.info()
print("\n" + "="*80 + "\n")

# Save cleaned dataset
df.to_csv('project_datasets/breast_cancer_wisconsin_diagnostic_cleaned.csv', index=False)
print("Cleaned dataset saved to: project_datasets/breast_cancer_wisconsin_diagnostic_cleaned.csv")