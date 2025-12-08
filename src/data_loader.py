# ================================================================
# data_loader.py 
# ================================================================

import pandas as pd
import os

def load_dataset1_raw(filepath):
    """
    Load Dataset 1: Breast Cancer Wisconsin (Original)
    No header issues.
    """
    return pd.read_csv(filepath)


def load_dataset2_raw(filepath):
    """
    Load Dataset 2: Breast Cancer Coimbra
    Expected columns include 'Age', 'BMI', ..., 'Classification'
    """
    df = pd.read_csv(filepath)
    return df


def load_dataset3_raw(filepath):
    """
    Load Dataset 3: Breast Cancer Wisconsin (Diagnostic)
    """
    return pd.read_csv(filepath)


def load_dataset4_raw(filepath):
    """
    Load Dataset 3: CDC Wonder Cancer Statistics (Incidence, Age Groups)
    """
    return pd.read_csv(filepath)
