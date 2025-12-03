
# ================================================================
# data_preprocessor.py 
# ================================================================

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
# -----------------------------
# Dataset 1 Cleaning
# -----------------------------
def clean_dataset1(df):
    """
    Clean Dataset 1 (Wisconsin Original)
    Steps:
    - Convert 'Bare Nuclei' to numeric
    - Drop NA
    - Drop duplicates
    - Remove sample ID column
    - Map Diagnosis (2->Benign->0, 4->Malignant->1)
    """

    # Clean Bare Nuclei
    df['Bare Nuclei'] = pd.to_numeric(df['Bare Nuclei'], errors='coerce')

    df = df.dropna()
    df = df.drop_duplicates()

    # Remove ID if present
    if 'Sample code number' in df.columns:
        df = df.drop(columns=['Sample code number'])

    # Rename column
    if 'Class' in df.columns:
        df = df.rename(columns={'Class': 'Diagnosis'})
    if 'diagnosis' in df.columns:
        df = df.drop(columns=['diagnosis'])
    # Binary target — (2=Benign, 4=Malignant)
    df['Is_Malignant'] = (df['Diagnosis'] == 4).astype(int)
    

    return df


# -----------------------------
# Dataset 2 Cleaning
# -----------------------------
def clean_dataset2(df):
    """
    Clean Dataset 2 (Coimbra)
    Original target is Classification:
        1 = Healthy
        2 = Patient (malignant)
    
    Final mapping should be:
        0 = Healthy
        1 = Malignant
    (Match Dataset1 & Dataset3)
    """

    if 'Classification' in df.columns:
        # EXACTLY same as your teammate’s cleaning script
        df['diagnosis'] = (df['Classification'] == 2).astype(int)
        df = df.drop(columns=['Classification'])

    # Drop duplicates (consistent with teammate)
    df = df.drop_duplicates()

    return df



# -----------------------------
# Dataset 3 Cleaning
# -----------------------------
def clean_dataset3(df):
    """
    Clean Dataset 3 (Wisconsin Diagnostic)
    Columns:
        id, diagnosis, radius_mean, ... , Unnamed: 32
    """

    cols_to_drop = ['id', 'Unnamed: 32']
    df = df.drop(columns=[c for c in cols_to_drop if c in df.columns])

    # Map diagnosis B/M → 0/1
    if 'diagnosis' in df.columns:
        df['diagnosis'] = df['diagnosis'].map({'B': 0, 'M': 1}).astype(int)

    return df


# -----------------------------
# Standardization Utility
# -----------------------------
def normalize_features(df, target_col='diagnosis'):
    """
    StandardScaler normalization for numerical features.
    """
    features = df.drop(columns=[target_col])
    target = df[target_col]

    scaler = StandardScaler()
    scaled = scaler.fit_transform(features)

    df_scaled = pd.DataFrame(scaled, columns=features.columns)
    df_scaled[target_col] = target.values

    return df_scaled
