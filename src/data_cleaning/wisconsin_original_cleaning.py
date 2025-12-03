import os
import kagglehub
import pandas as pd

os.environ['KAGGLEHUB_CACHE'] = '/Users/juliana_miller/Desktop/ece143-assignments/project/project_datasets'

path = kagglehub.dataset_download("mariolisboa/breast-cancer-wisconsin-original-data-set")

print("Path to dataset files:", path)

df = pd.read_csv('project_datasets/breast_cancer_bd.csv')

# Take a loot at the dataset
print("\n" + "="*80 + "\n")
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

# 'Bare Nuclei' column seems to be of type object, appears to be '?' in some rows
# Changed 'Bare Nuclei' column to float to change '?' to 'NaN', dropped rows with missing values, and eliminated duplicates
df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'], errors='coerce')
print("\nMissing values in Bare Nuclei Column:", df['Bare Nuclei'].isna().sum())
df = df.dropna()
duplicates = df.duplicated().sum()
print(f"\nDuplicates found: {duplicates}")
df = df.drop_duplicates()
print("\n" + "="*80 + "\n")
# print("Dataset info (column names and types):") 
# df.info()

# Removed 'Sample code number' column as it didn't serve a purpose for us
# Renamed 'Class' column to 'Diagnosis' as it is more intuitive 
# Added 'Is_Malignant column with 1=malignant and 0=benign, possibly easier to work with as class column uses 2=benign and 4=malignant
df = df.drop('Sample code number', axis=1)
df = df.rename(columns={'Class': 'Diagnosis'})
df['Is_Malignant'] = (df['Diagnosis'] == 4).astype(int)
print("\n" + "="*80 + "\n")
# print("Dataset info (column names and types):")
# df.info()

feature_cols = ['Clump Thickness', 'Uniformity of Cell Size', 
                'Uniformity of Cell Shape', 'Marginal Adhesion',
                'Single Epithelial Cell Size', 'Bare Nuclei',
                'Bland Chromatin', 'Normal Nucleoli', 'Mitoses']

print("\nFeature ranges:")
print(df[feature_cols].describe())

# Dataset uses standardized 1-10 scale, checking to make sure all values are valid
for col in feature_cols:
    col_range = df[(df[col] < 1) | (df[col] > 10)]
    if len(col_range) > 0:
        print(f"\n{col} has {len(col_range)} values outside 1-10 range")
print("\n" + "="*80 + "\n")

print("Cleaned dataset:")
print("Cleaned shape:", df.shape)
print("\n" + "="*80 + "\n")
print("Cleaned Dataset info (column names and types):")
df.info()
print("\n" + "="*80 + "\n")

# Save clean dataset
df.to_csv('project/project_datasets/breast_cancer_wisconsin_original_cleaned.csv', index=False)
print("Cleaned dataset saved to: project/project_datasets/breast_cancer_wisconsin_original_cleaned.csv")