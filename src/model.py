"""
Model training and evaluation utilities.
"""
import numpy as np
import joblib
from pathlib import Path
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
)

MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def evaluate_model(model, X_test, y_test, model_name="Model"):
    """Print classification metrics and return results dict."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    print(f"\n{'='*50}")
    print(f"  {model_name} Evaluation")
    print(f"{'='*50}")
    print(classification_report(y_test, y_pred, target_names=["Fully Paid", "Default"]))

    results = {
        "confusion_matrix": confusion_matrix(y_test, y_pred),
        "y_pred": y_pred,
    }

    if y_prob is not None:
        auc = roc_auc_score(y_test, y_prob)
        print(f"ROC AUC: {auc:.4f}")
        results["roc_auc"] = auc
        results["y_prob"] = y_prob

    return results


def save_model(model, filename):
    """Save model to disk."""
    filepath = MODELS_DIR / filename
    joblib.dump(model, filepath)
    print(f"Model saved to {filepath}")


def load_model(filename):
    """Load model from disk."""
    filepath = MODELS_DIR / filename
    return joblib.load(filepath)
