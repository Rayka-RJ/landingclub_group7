"""
Data loading and initial cleaning utilities for Lending Club dataset.
"""
import pandas as pd
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DATA_DIR = PROJECT_ROOT / "data" / "processed"


def load_accepted_loans(nrows=None):
    """Load accepted loans dataset."""
    filepath = RAW_DATA_DIR / "accepted.csv"
    df = pd.read_csv(filepath, low_memory=False, nrows=nrows)
    return df


def load_rejected_loans(nrows=None):
    """Load rejected loans dataset."""
    filepath = RAW_DATA_DIR / "rejected.csv"
    df = pd.read_csv(filepath, low_memory=False, nrows=nrows)
    return df


def save_processed(df, filename):
    """Save processed dataframe to parquet."""
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
    filepath = PROCESSED_DATA_DIR / filename
    df.to_parquet(filepath, index=False)
    print(f"Saved to {filepath}")


def load_processed(filename):
    """Load processed dataframe from parquet."""
    filepath = PROCESSED_DATA_DIR / filename
    return pd.read_parquet(filepath)
