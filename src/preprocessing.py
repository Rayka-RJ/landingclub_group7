"""
Feature engineering and preprocessing pipeline.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder


def define_target(df, col="loan_status"):
    """
    Create binary target: 1 = default/bad loan, 0 = fully paid/good loan.
    Keeps only clearly resolved loans (removes 'Current', 'In Grace Period', etc.)
    """
    bad_status = [
        "Charged Off",
        "Default",
        "Late (31-120 days)",
        "Late (16-30 days)",
        "Does not meet the credit policy. Status:Charged Off",
    ]
    good_status = [
        "Fully Paid",
        "Does not meet the credit policy. Status:Fully Paid",
    ]
    df = df[df[col].isin(bad_status + good_status)].copy()
    df["target"] = df[col].apply(lambda x: 1 if x in bad_status else 0)
    return df


def drop_leakage_columns(df):
    """Remove columns that leak future information or are not useful."""
    leak_cols = [
        "funded_amnt", "funded_amnt_inv", "total_pymnt", "total_pymnt_inv",
        "total_rec_prncp", "total_rec_int", "total_rec_late_fee",
        "recoveries", "collection_recovery_fee", "last_pymnt_d",
        "last_pymnt_amnt", "last_credit_pull_d", "last_fico_range_high",
        "last_fico_range_low", "out_prncp", "out_prncp_inv",
        "pymnt_plan", "url", "desc", "title", "id", "member_id",
        "policy_code", "application_type",
    ]
    existing = [c for c in leak_cols if c in df.columns]
    return df.drop(columns=existing)


def basic_feature_engineering(df):
    """Create commonly used derived features."""
    if "issue_d" in df.columns:
        df["issue_d"] = pd.to_datetime(df["issue_d"], format="mixed", errors="coerce")
        df["issue_year"] = df["issue_d"].dt.year
        df["issue_month"] = df["issue_d"].dt.month

    if "earliest_cr_line" in df.columns:
        df["earliest_cr_line"] = pd.to_datetime(
            df["earliest_cr_line"], format="mixed", errors="coerce"
        )
        if "issue_d" in df.columns:
            df["credit_history_years"] = (
                (df["issue_d"] - df["earliest_cr_line"]).dt.days / 365.25
            )

    if "term" in df.columns:
        df["term_months"] = (
            df["term"].astype(str).str.extract(r"(\d+)").astype(float)
        )

    return df
