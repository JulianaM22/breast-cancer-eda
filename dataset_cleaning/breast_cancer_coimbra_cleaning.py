import os
import kagglehub
import pandas as pd

os.environ['KAGGLEHUB_CACHE'] = '/Users/juliana_miller/Desktop/ece143-assignments/project/project_datasets'

path = kagglehub.dataset_download("atom1991/breast-cancer-coimbra")

print("Path to dataset files:", path)

df = pd.read_csv('project_datasets/Coimbra_breast_cancer_dataset.csv')

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

# Removed duplicates
duplicates = df.duplicated().sum()
print(f"\nDuplicates found: {duplicates}")
df = df.drop_duplicates()
print("\n" + "="*80 + "\n")

# Changed 'Classification' column values from 1=healthy 2=breast cancer patient (malignant) to 0=healthy 1=patient/malignant to match other dataset
df['Classification'] = (df['Classification'] == 2).astype(int)
print(f"Patients (Malignant): {df['Classification'].sum()}")
print(f"Healthy (Control): {(df['Classification'] == 0).sum()}")
print("\n" + "="*80 + "\n")

# Get all feature columns except for 'Classification' (different scale)
feature_cols = [col for col in df.columns if col != 'Classification']

print("Feature ranges:")
print(df[feature_cols].describe())
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
df.to_csv('project_datasets/breast_cancer_coimbra_cleaned.csv', index=False)
print("Cleaned dataset saved to: project_datasets/breast_cancer_coimbra_cleaned.csv")