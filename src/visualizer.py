# ================================================================
# visualizer.py 
# ================================================================

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import shap
from sklearn.ensemble import RandomForestClassifier

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

    plt.title("Dataset 3 — Malignancy by Radius Group")
    plt.xticks(rotation=0)
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
