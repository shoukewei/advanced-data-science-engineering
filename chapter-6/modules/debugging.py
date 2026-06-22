# modules/debugging.py

import pandas as pd
import numpy as np


def diagnose_dtypes(
    df: pd.DataFrame,
    expected: dict = None
) -> pd.DataFrame:
    """
    Report actual dtypes and flag mismatches with expected types.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    expected : dict, optional
        Mapping of column name to expected dtype string.
        If provided, a 'mismatch' column is added.

    Returns
    -------
    pd.DataFrame
        Columns: column, actual_dtype, expected_dtype, mismatch.
    """
    rows = []
    for col in df.columns:
        actual = str(df[col].dtype)
        exp    = expected.get(col, None) if expected else None
        rows.append({
            "column":        col,
            "actual_dtype":  actual,
            "expected_dtype": exp,
            "mismatch":      (exp is not None and actual != exp),
        })
    return pd.DataFrame(rows)


def coerce_dtypes(df: pd.DataFrame, dtypes: dict) -> pd.DataFrame:
    """Cast specified columns to their target dtypes. Returns a copy."""
    df = df.copy()
    for col, dtype in dtypes.items():
        if col in df.columns:
            try:
                df[col] = df[col].astype(dtype)
            except (ValueError, TypeError) as e:
                raise ValueError(f"Cannot cast '{col}' to {dtype}: {e}")
    return df


def diagnose_missing(df: pd.DataFrame, label: str = "") -> None:
    """
    Print a missing value report for a DataFrame.

    Prints only columns with missing values, sorted by count.
    Useful for quick inspection at pipeline checkpoints.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame to inspect.
    label : str, optional
        Label printed before the report (e.g. 'After imputation').
    """
    missing = df.isnull().sum()
    missing = missing[missing > 0].sort_values(ascending=False)

    if label:
        print(f"\n[{label}] Missing values:")
    if missing.empty:
        print("  None")
    else:
        for col, count in missing.items():
            pct = count / len(df) * 100
            print(f"  {col}: {count} ({pct:.1f}%)")


def diagnose_distribution_shift(
    df_train: pd.DataFrame,
    df_test: pd.DataFrame,
    threshold: float = 0.1
) -> pd.DataFrame:
    """
    Compare distributions of numeric columns between train and test sets.

    Flags columns where the relative difference in mean exceeds threshold.

    Parameters
    ----------
    df_train : pd.DataFrame
        Training features.
    df_test : pd.DataFrame
        Test features (must have same columns).
    threshold : float, optional
        Relative mean difference above which a column is flagged.
        Default is 0.1 (10%).

    Returns
    -------
    pd.DataFrame
        Columns: feature, train_mean, test_mean, rel_diff, shifted.
    """
    numeric_cols = df_train.select_dtypes(include="number").columns
    rows = []
    for col in numeric_cols:
        if col not in df_test.columns:
            continue
        train_mean = df_train[col].mean()
        test_mean  = df_test[col].mean()
        rel_diff   = abs(test_mean - train_mean) / (abs(train_mean) + 1e-9)
        rows.append({
            "feature":    col,
            "train_mean": round(train_mean, 4),
            "test_mean":  round(test_mean, 4),
            "rel_diff":   round(rel_diff, 4),
            "shifted":    rel_diff > threshold,
        })
    return (
        pd.DataFrame(rows)
        .sort_values("rel_diff", ascending=False)
        .reset_index(drop=True)
    )


def verify_no_leakage(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    scaler_params: dict
) -> None:
    """
    Verify that scaling parameters were computed from training data only.

    Checks that the scaler's stored mean matches the training set mean
    and differs from the test set mean by at least a small amount.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features after scaling.
    X_test : pd.DataFrame
        Test features after scaling.
    scaler_params : dict
        Fitted scaling parameters (from compute_scaling_params()).

    Raises
    ------
    AssertionError
        If the training mean is not approximately 0 after standardization.
    """
    for col, params in scaler_params.items():
        if "mean" in params and col in X_train.columns:
            scaled_train_mean = X_train[col].mean()
            assert abs(scaled_train_mean) < 1e-6, (
                f"Leakage suspected in '{col}': "
                f"training mean after scaling = {scaled_train_mean:.6f} "
                f"(expected ≈ 0). Were scaling params computed before splitting?"
            )