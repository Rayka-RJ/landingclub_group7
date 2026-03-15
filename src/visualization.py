"""
Plotting utilities for EDA and model evaluation.
"""
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from pathlib import Path

FIGURES_DIR = Path(__file__).resolve().parent.parent / "figures"
FIGURES_DIR.mkdir(exist_ok=True)


def plot_target_distribution(y, title="Loan Status Distribution", save=True):
    """Plot bar chart of target variable distribution."""
    fig, ax = plt.subplots(figsize=(6, 4))
    labels = ["Fully Paid", "Default"]
    counts = [sum(y == 0), sum(y == 1)]
    colors = ["#2ecc71", "#e74c3c"]
    ax.bar(labels, counts, color=colors)
    ax.set_title(title)
    ax.set_ylabel("Count")
    for i, (label, count) in enumerate(zip(labels, counts)):
        pct = count / sum(counts) * 100
        ax.text(i, count, f"{pct:.1f}%", ha="center", va="bottom")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "target_distribution.png", dpi=150)
    return fig


def plot_correlation_matrix(df, top_n=20, save=True):
    """Plot correlation heatmap for top N numeric features."""
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] > top_n:
        # Select features with highest variance
        top_cols = numeric_df.var().nlargest(top_n).index
        numeric_df = numeric_df[top_cols]

    fig, ax = plt.subplots(figsize=(12, 10))
    corr = numeric_df.corr()
    sns.heatmap(corr, annot=False, cmap="RdBu_r", center=0, ax=ax)
    ax.set_title("Feature Correlation Matrix")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "correlation_matrix.png", dpi=150)
    return fig


def plot_feature_importance(feature_names, importances, top_n=20, save=True):
    """Plot horizontal bar chart of feature importances."""
    idx = np.argsort(importances)[-top_n:]
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.barh(range(len(idx)), importances[idx], color="#3498db")
    ax.set_yticks(range(len(idx)))
    ax.set_yticklabels([feature_names[i] for i in idx])
    ax.set_xlabel("Importance")
    ax.set_title(f"Top {top_n} Feature Importances")
    plt.tight_layout()
    if save:
        fig.savefig(FIGURES_DIR / "feature_importance.png", dpi=150)
    return fig
