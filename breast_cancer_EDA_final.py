import sys
sys.path.append('./src')

import pandas as pd
import numpy as np
import warnings
warnings.filterwarnings('ignore')

import src.data_loader as data_loader
import src.data_preprocessor as data_preprocessor
import src.visualizer as viz

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

print("All modules loaded successfully!")

from IPython.display import Markdown

Markdown("""
# Final Presentation Notebook  
### Group 11 — ECE143 Data Science in Practice  
This notebook consolidates **all datasets, all EDA, and all visualizations** using the updated visualizer structure.

### Datasets:
- **Dataset 1:** Wisconsin Breast Cancer (Original)
- **Dataset 2:** Coimbra Breast Cancer (Metabolic)
- **Dataset 3:** Wisconsin Breast Cancer (Diagnostic Imaging)

All plots use:
- Cleaned and unified target variables  
- Updated palette (0 = benign/healthy, 1 = malignant/patient)  
- Dataset-specific functions  
""")

# ------- Dataset 1 ---------

raw_df1 = data_loader.load_dataset1_raw("dataset/breast_cancer_bd.csv")
df1 = data_preprocessor.clean_dataset1(raw_df1)
df1.head()

print(df1.describe())
df1['Is_Malignant'].value_counts()

viz.plot_d1_anova(df1)
viz.plot_d1_boxplots(df1)

X1 = df1.drop(columns=["Is_Malignant"])
y1 = df1["Is_Malignant"]

rf1 = RandomForestClassifier(random_state=42)
rf1.fit(X1, y1)

viz.plot_d1_rf_importance(df1)

# ------- Dataset 2 ---------

raw_df2 = data_loader.load_dataset2_raw("dataset/Coimbra_breast_cancer_dataset.csv")
df2 = data_preprocessor.clean_dataset2(raw_df2)
df2.head()

print(df2.describe())
df2['diagnosis'].value_counts()

viz.plot_d2_class_distribution(df2)
viz.plot_d2_kde(df2)
viz.plot_d2_anova(df2)

# ------- Dataset 3 ---------

# Dataset 3 – Prepare standardized features (matching teammate’s code)
raw_df3 = data_loader.load_dataset3_raw("dataset/breast_cancer_dia.csv")
df3 = data_preprocessor.clean_dataset3(raw_df3)

# Split features & target
features = df3.drop(columns=["diagnosis"])
target = df3["diagnosis"].astype(int)

# Standardize (z-score)
means = features.mean()
stds = features.std()
features_normalized = (features - means) / stds

# Final X and y for modeling
X3 = features_normalized
y3 = target

df3.head()

viz.plot_d3_radius_group(df3)
viz.plot_d3_concavepoints_group(df3)

viz.run_d3_model_comparison(df3)
viz.plot_d3_shap(df3)

# ------- Dataset 4 ---------

raw_df4 = data_loader.load_dataset4_raw("dataset/USCancerIncidence1999_2022.csv")
df4 = data_preprocessor.clean_dataset4(raw_df4)
df4.head()

print(df4.describe())

viz.plot_d4_age_groups(df4)

# ------- Datasets 1-3 ---------

viz.plot_all_feature_importance(df1, df2, df3)
