
<div align="center"> 
  <img src="other_resources/UCSD_logo.svg" width="30%" />
</div>

<h3 align="center">An EDA on Malignant Breast Cancer Cells</h3>

---

## Overview

Breast cancer is one of the most prevalent cancers affecting women worldwide and remains a major contributor to cancer-related mortality. According to WHO (2022), it ranks second in global incidence and is the leading cause of cancer death among women. In the United States, it is surpassed only by lung cancer (CDC, 2025).

This project conducts exploratory data analysis (EDA) on three breast cancer datasets to investigate statistical, biochemical, and morphological factors associated with malignant versus benign tumors. Using Python-based visualizations and statistical techniques, we identify patterns and feature relationships relevant to malignancy.

### Goals
- Examine how malignancy varies across demographic, metabolic, and morphological features  
- Identify features that most effectively distinguish malignant from benign tumors  
- Visualize data patterns through distribution plots, correlation maps, and model-based feature importance  

---

## Datasets Analyzed

1. **Dataset 1 — Wisconsin Breast Cancer (Original)**  
   Cytological FNA cell measurements (9 features)

2. **Dataset 2 — Coimbra Breast Cancer (Metabolic)**  
   Blood and metabolic biomarkers (10 features)

3. **Dataset 3 — Wisconsin Breast Cancer (Diagnostic)**  
   Imaging-based morphology and texture features (30 features)

All datasets follow unified cleaning, feature processing, and visualization pipelines.

---

## Repository Structure

```txt
BREAST-CANCER-EDA-MAIN/
│
├── datasets/
│   ├── cleaned_datasets/
│   ├── normalized_datasets/
│   ├── original_datasets/
│   └── README.md
│
├── docs/
│   ├── [Project Proposal] Malignant Breast Cancer Cells.pdf
│   ├── Dataset1_breast_cancer_wisconsin_original.pdf
│   ├── Dataset2_Coimbra.pdf
│   ├── Dataset3_Breast Cancer Wisconsin (Diagnostic).pdf
│   ├── Breast_cancer_EDA_final.pdf
│
├── notebooks/
│   ├── Dataset1_breast_cancer_wisconsin_original.ipynb
│   ├── Dataset2_Coimbra.ipynb
│   ├── Dataset3_Breast Cancer Wisconsin (Diagnostic).ipynb
│   └── Breast_cancer_EDA_final.ipynb
│
├── src/
│   ├── data_loader.py
│   ├── data_preprocessor.py
│   ├── visualizer.py
│   └── data_cleaning/
│
├── results/
│   └── (auto-generated figures and model outputs)
│
├── other_resources/
│   └── UCSD_logo.svg
│
├── README.md
└── .gitignore
````

---

## Core Modules

### `data_loader.py`

* Centralized dataset loading
* Standardized paths for raw and cleaned versions

### `data_preprocessor.py`

* Type corrections, label cleaning, missing value handling
* Z-score normalization
* Dataset-specific preprocessing logic

### `visualizer.py`

Provides the complete visualization suite:

* Class distribution
* KDE + histogram overlays
* Boxplots & violin plots
* Correlation heatmaps
* Mutual information & ANOVA F-test
* Random Forest feature importance
* SHAP interpretability
* Dataset-specific wrappers (`plot_d1_*`, `plot_d2_*`, `plot_d3_*`)

---

## Modeling (Dataset 3)

Includes 10-fold cross-validated performance comparison for:

* Logistic Regression
* KNN
* SVM
* Naive Bayes
* Decision Tree
* Random Forest
* Gradient Boosting

Also includes:

* SHAP summary plots
* Random Forest feature importance ranking

---

## How to Run the Project

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Launch the notebook environment

```bash
jupyter notebook
```
---

## Third-Party Modules Used

### Scientific Computing

* numpy
* pandas
* scipy

### Visualization

* matplotlib
* seaborn

### Machine Learning (scikit-learn)

* Logistic Regression
* KNN
* SVM
* Naive Bayes
* Decision Tree
* Random Forest
* Gradient Boosting
* StandardScaler
* StratifiedKFold
* train_test_split

### Model Interpretability

* shap

### Utilities

* tabulate

*All dependencies are included in `requirements.txt`.*

---

## Major Features Implemented

* Unified preprocessing pipeline
* Class distribution plots
* KDE / distribution plots
* Boxplots & violin plots
* Correlation heatmaps
* Outlier visualization
* Mutual Information + ANOVA scoring (Dataset 2)
* Random Forest feature importance
* SHAP global interpretability (Dataset 3)
* 10-fold model comparison for Dataset 3

---

