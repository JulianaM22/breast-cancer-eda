# ================================================================
# visualizer.py 
# ================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import f_classif

import data_loader, data_preprocessor

sns.set(style="whitegrid")

# Color maps for binary labels
PALETTE_BIN = {
    "0": "#3498db",
    "1": "#e74c3c"
}

LABELS_BIN = {
    "0": "Benign / Healthy",
    "1": "Malignant / Patient"
}

# Convert label column to string type
def _ensure_str_labels(df, col):
    df = df.copy()
    df[col] = df[col].astype(str)
    return df


# ================================================================
# Dataset 1
# ================================================================

def plot_d1_heatmap(df):
    """Correlation heatmap for Dataset 1."""
    df = df.copy()
    drop_cols = ["Diagnosis"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    corr = df.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True)
    plt.title("Dataset 1 — Correlation Heatmap")
    plt.tight_layout()
    plt.show()


def plot_d1_feature_ranking(df):
    """Feature–target correlations for Dataset 1."""
    df = df.copy()
    target = "Is_Malignant"
    drop_cols = ["Diagnosis"]
    for col in drop_cols:
        if col in df.columns:
            df = df.drop(columns=[col])

    corr = df.corr(numeric_only=True)[target].drop(target).sort_values(ascending=False)

    plt.figure(figsize=(10, 6))
    sns.barplot(x=corr.values, y=corr.index, palette='viridis')
    plt.title("Dataset 1 — Feature–Malignancy Correlation")
    plt.xlabel("Correlation with Is_Malignant")
    plt.tight_layout()
    plt.show()


def plot_d1_boxplots(df):
    """Boxplots for Dataset 1 features split by class."""
    df = _ensure_str_labels(df, "Is_Malignant")
    features = [c for c in df.columns if c not in ["Is_Malignant", "Diagnosis"]]

    melted = df.melt(id_vars="Is_Malignant", value_vars=features,
                     var_name="Feature", value_name="Value")

    plt.figure(figsize=(16, 8))
    sns.boxplot(
        data=melted, x="Feature", y="Value", hue="Is_Malignant",
        palette=PALETTE_BIN
    )
    plt.title("Dataset 1 — Feature Distributions")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()


def plot_d1_rf_importance(df):
    """Random Forest feature importance for Dataset 1."""
    df = df.copy()
    target = "Is_Malignant"

    X = df.drop(columns=["Is_Malignant", "Diagnosis"], errors="ignore")
    y = df[target]

    rf = RandomForestClassifier(random_state=42)
    rf.fit(X, y)

    importances = pd.Series(rf.feature_importances_, index=X.columns).sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title("Dataset 1 — Random Forest Importances")
    plt.tight_layout()
    plt.show()

def plot_d1_anova(df):
    """ANOVA F-test for Dataset 1."""
    feature_cols = [c for c in df.columns if c not in ['Is_Malignant', 'Diagnosis']]
    X = df[feature_cols].values
    y = df['Is_Malignant'].values
    
    f_vals, _ = f_classif(X, y)
    
    f_df = pd.DataFrame({
        "feature": feature_cols,
        "f_value": f_vals,
    }).sort_values("f_value", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=f_df, y='feature', x='f_value', palette='viridis')
    plt.title("Dataset 1 — ANOVA F-test Scores")
    plt.xlabel("F-value")
    plt.tight_layout()
    plt.show()

# ================================================================
# Dataset 2
# ================================================================

def plot_d2_hist(df):
    """Histograms for Dataset 2 features."""
    features = [c for c in df.columns if c != "diagnosis"]

    n_cols = 3
    n_rows = (len(features) + 2) // 3

    plt.figure(figsize=(15, 4*n_rows))
    for i, col in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.histplot(df[col], bins=20, kde=True, color='steelblue')
        plt.title(col)
    plt.tight_layout()
    plt.show()


def plot_d2_class_distribution(df):
    """Class counts for Dataset 2."""
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].astype(str)

    plt.figure(figsize=(6, 4))
    sns.countplot(data=df, x="diagnosis", palette=PALETTE_BIN)
    plt.title("Dataset 2 — Class Distribution")
    plt.xlabel("Diagnosis")
    plt.ylabel("Count")
    plt.xticks([0,1], [" Healthy", " Patient"])
    plt.tight_layout()
    plt.show()


def plot_d2_kde(df):
    """KDE curves for Dataset 2 features."""
    df = _ensure_str_labels(df, "diagnosis")
    features = [c for c in df.columns if c != "diagnosis"]

    n_cols = 3
    n_rows = (len(features) + 2) // 3

    plt.figure(figsize=(16, 4*n_rows))
    for i, col in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        for label, color in PALETTE_BIN.items():
            subset = df[df["diagnosis"] == label]
            sns.kdeplot(subset[col], label=LABELS_BIN[label], color=color)
        plt.title(col)
        plt.legend()
    plt.tight_layout()
    plt.show()


def plot_d2_violin(df):
    """Violin plots for Dataset 2."""
    df = _ensure_str_labels(df, "diagnosis")
    features = [c for c in df.columns if c != "diagnosis"]

    n_cols = 3
    n_rows = (len(features) + 2) // 3
    plt.figure(figsize=(16, 5*n_rows))

    for i, col in enumerate(features, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.violinplot(
            data=df, x="diagnosis", y=col,
            palette=PALETTE_BIN, inner="quartile"
        )
        plt.title(col)

    plt.tight_layout()
    plt.show()


def plot_d2_heatmap(df):
    """Correlation heatmap for Dataset 2."""
    df2_corr = df.corr(numeric_only=True)
    plt.figure(figsize=(10, 8))
    sns.heatmap(df2_corr, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Dataset 2 — Correlation Heatmap")
    plt.tight_layout()
    plt.show()

def plot_d2_anova(df):
    """ANOVA F-test for Dataset 2."""
    feature_cols = [c for c in df.columns if c != 'diagnosis']
    X = df[feature_cols].values
    y = df['diagnosis'].values
    
    f_vals, _ = f_classif(X, y)
    
    f_df = pd.DataFrame({
        "feature": feature_cols,
        "f_value": f_vals,
    }).sort_values("f_value", ascending=False)
    
    plt.figure(figsize=(10, 6))
    sns.barplot(data=f_df, y='feature', x='f_value', palette='viridis')
    plt.title("Dataset 2 — ANOVA F-test Scores")
    plt.xlabel("F-value")
    plt.tight_layout()
    plt.show()

def plot_d2_age_distribution(df):
    """Bar chart of malignancy by age groups for Dataset 2."""
    df = df.copy()
    
    bins = [0, 40, 50, 60, 70, 100]
    labels = ['<40', '40-49', '50-59', '60-69', '70+']
    df['age_group'] = pd.cut(df['Age'], bins=bins, labels=labels)
    
    rates = df.groupby('age_group')['diagnosis'].apply(lambda x: (x==1).sum()/len(x)*100)
    
    plt.figure(figsize=(8, 6))
    plt.bar(rates.index, rates.values, color='coral')
    plt.title("Dataset 2 — Malignancy Rate by Age Group")
    plt.ylabel("% Malignant")
    plt.xlabel("Age Group")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ================================================================
# Dataset 3
# ================================================================

def plot_d3_mean_heatmap(df):
    """Correlation heatmap for mean features in Dataset 3."""
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    df_mean = df[mean_cols + ["diagnosis"]]

    corr = df_mean.corr(numeric_only=True)

    plt.figure(figsize=(12, 10))
    sns.heatmap(corr, annot=True, fmt='.2f',
                cmap='coolwarm', center=0, square=True)
    plt.title("Dataset 3 — Mean Feature Correlation")
    plt.tight_layout()
    plt.show()


def plot_d3_regression_mean(df):
    """Regression plots for mean features in Dataset 3."""
    mean_cols = [
        "radius_mean", "texture_mean", "perimeter_mean", "area_mean",
        "smoothness_mean", "compactness_mean", "concavity_mean",
        "concave points_mean", "symmetry_mean", "fractal_dimension_mean"
    ]
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].astype(int)

    n_cols = 3
    n_rows = (len(mean_cols) + 2) // 3

    plt.figure(figsize=(18, 5*n_rows))
    for i, col in enumerate(mean_cols, 1):
        plt.subplot(n_rows, n_cols, i)
        sns.regplot(
            x=df[col], y=df["diagnosis"],
            scatter_kws={"s": 10, "alpha": 0.6},
            line_kws={"color": "steelblue"}
        )
        plt.title(f"{col} vs diagnosis")

    plt.tight_layout()
    plt.show()


def plot_d3_radius_group(df):
    """Stacked bar chart by radius groups."""
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].astype(str)

    bins = [0, 10, 15, 20, 30]
    labels = ["<10", "10–15", "15–20", "≥20"]

    df["radius_group"] = pd.cut(df["radius_mean"], bins=bins, labels=labels)
    counts = df.groupby(["radius_group", "diagnosis"]).size().unstack(fill_value=0)

    counts.plot(
        kind='bar', stacked=True, figsize=(10, 6),
        color=[PALETTE_BIN["0"], PALETTE_BIN["1"]]
    )

    plt.title("Malignancy by Radius Group")
    plt.xlabel("Radius")
    plt.ylabel("Number of Diagnoses")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

def plot_d3_concavepoints_group(df):
    """Stacked bar chart by concave points groups."""
    df = df.copy()
    df["diagnosis"] = df["diagnosis"].astype(str)

    bins = [0.0, 0.05, 0.1, 0.15, 0.2]
    labels = ["<0.05", "0.05-0.1", "0.1-0.15", "≥0.2"]

    df["concave_points_group"] = pd.cut(df["concave points_mean"], bins=bins, labels=labels)
    counts = df.groupby(["concave_points_group", "diagnosis"]).size().unstack(fill_value=0)

    counts.plot(
        kind='bar', stacked=True, figsize=(10, 6),
        color=[PALETTE_BIN["0"], PALETTE_BIN["1"]]
    )

    plt.title("Malignancy by Concave Points Group")
    plt.xlabel("Concave Points")
    plt.ylabel("Number of Diagnoses")
    plt.xticks(rotation=0)
    plt.tight_layout()
    plt.show()

# ================================================================
# Dataset 4
# ================================================================

def plot_d4_age_groups(df):
    """Line chart for age groups over time in Dataset 4."""
    
    young_groups = ['15-19 years', '20-24 years', '25-29 years', 
                      '30-34 years', '35-39 years', '40-44 years', '45-49 years']
    old_groups = ['50-54 years', '55-59 years', '60-64 years', 
                    '65-69 years', '70-74 years', '75-79 years', '80-84 years', '85+ years']
    
    df_plot = df[df['Age Groups'].isin(young_groups + old_groups)].copy()
    
    def categorize_age(age_str):
        if age_str in young_groups:
            if any(x in age_str for x in ['15-19', '20-24', '25-29', '30-34', '35-39']):
                return '<40'
            else:
                return '40-49'
        elif age_str in old_groups:
            if any(x in age_str for x in ['50-54', '55-59']):
                return '50-59'
            elif any(x in age_str for x in ['60-64', '65-69']):
                return '60-69'
            else:
                return '70+'
        return 'Other'
    
    df_plot['Age Category'] = df_plot['Age Groups'].apply(categorize_age)
    
    trends = df_plot.groupby(['Year', 'Age Category'])['Count'].sum().reset_index()
    
    pivot_count = trends.pivot(index='Year', columns='Age Category', values='Count')
    pivot_count = pivot_count[['<40', '40-49', '50-59', '60-69', '70+']]  # Order columns
    
    plt.figure(figsize=(14, 7))

    colors = ['#e74c3c', '#f39c12', '#3498db', '#2ecc71', '#9b59b6']
    
    for i, col in enumerate(pivot_count.columns):
        plt.plot(pivot_count.index, pivot_count[col], 
                marker='o', label=col, linewidth=2.5, 
                color=colors[i], markersize=4)
    
    plt.title("Breast Cancer Incidence Trends by Age Group (1999-2022)", 
              fontsize=14, fontweight='bold')
    plt.xlabel("Year", fontsize=12)
    plt.ylabel("Number of Cases", fontsize=12)
    plt.legend(title="Age Group", fontsize=11)
    plt.grid(alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

# ================================================================
# SHAP + RF utilities
# ================================================================

def plot_rf_feature_importance(model, feature_names):
    """Random Forest feature importance."""
    importances = pd.Series(model.feature_importances_, index=feature_names).sort_values()

    plt.figure(figsize=(10, 6))
    sns.barplot(x=importances.values, y=importances.index, palette='viridis')
    plt.title("Random Forest Feature Importances")
    plt.tight_layout()
    plt.show()


def plot_shap_summary(model, X):
    """SHAP summary plot."""
    shap.initjs()
    explainer = shap.TreeExplainer(model)
    vals = explainer.shap_values(X)

    if isinstance(vals, list):
        shap.summary_plot(vals[1], X)
    else:
        shap.summary_plot(vals, X)


def plot_d3_shap(df):
    """SHAP summary for Dataset 3 mean features."""
    mean_cols = [c for c in df.columns if c.endswith("_mean")]
    X = df[mean_cols]
    y = df["diagnosis"].astype(int)

    model = RandomForestClassifier(random_state=42)
    model.fit(X, y)

    plot_shap_summary(model, X)

# ================================================================
# Combined Dataset 1-3 Feature Importance Comparison
# ================================================================

def plot_all_feature_importance(df1, df2, df3):
    """Bar chart of highest random forest feature importance values across all three datasets."""
    
    # Dataset 1
    X1 = df1.drop(columns=['Is_Malignant', 'Diagnosis'], errors='ignore')
    y1 = df1['Is_Malignant']
    rf1 = RandomForestClassifier(random_state=42, n_estimators=100)
    rf1.fit(X1, y1)
    imp1 = pd.Series(rf1.feature_importances_, index=X1.columns).nlargest(3)
    
    # Dataset 2
    X2 = df2.drop(columns=['diagnosis'])
    y2 = df2['diagnosis']
    rf2 = RandomForestClassifier(random_state=42, n_estimators=100)
    rf2.fit(X2, y2)
    imp2 = pd.Series(rf2.feature_importances_, index=X2.columns).nlargest(3)
    
    # Dataset 3
    mean_cols = [c for c in df3.columns if c.endswith('_mean')]
    X3 = df3[mean_cols]
    y3 = df3['diagnosis']
    rf3 = RandomForestClassifier(random_state=42, n_estimators=100)
    rf3.fit(X3, y3)
    imp3 = pd.Series(rf3.feature_importances_, index=X3.columns).nlargest(3)
    
    data = {
        'Feature': list(imp1.index) + list(imp2.index) + list(imp3.index),
        'Importance': list(imp1.values) + list(imp2.values) + list(imp3.values),
        'Dataset': ['Microscopy-based Morphology']*3 + ['Metabolic']*3 + ['Imaging-based morphology']*3
    }
    
    df_importance = pd.DataFrame(data)
    
    plt.figure(figsize=(12, 6))
    sns.barplot(data=df_importance, y='Feature', x='Importance', 
                hue='Dataset', palette=['#e13661', '#ff6f4b', '#a11477'])
    plt.title("Discriminative Features Across Diagnostic Approaches")
    plt.xlabel("Random Forest Feature Importance")
    plt.tight_layout()
    plt.show()