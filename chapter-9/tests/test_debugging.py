import pandas as pd
import pytest
from modules.debugging import (
    diagnose_dtypes, coerce_dtypes,
    diagnose_missing, diagnose_distribution_shift,
    verify_no_leakage,
)


def test_diagnose_dtypes_flags_mismatch():
    df = pd.DataFrame({"a": [1, 2], "b": ["x", "y"]})
    expected = {"a": "int64", "b": "float64"}
    rpt = diagnose_dtypes(df, expected)
    assert "b" in rpt.loc[rpt["mismatch"], "column"].tolist()


def test_coerce_dtypes_casts_columns():
    df = pd.DataFrame({"a": ["1", "2"], "b": ["3.0", "4.1"]})
    out = coerce_dtypes(df, {"a": "int64", "b": "float64"})
    assert out["a"].dtype == "int64"
    assert out["b"].dtype == "float64"


def test_diagnose_missing_reports(capsys):
    df = pd.DataFrame({"x": [1, None, 3], "y": [None, None, 2]})
    diagnose_missing(df, label="after")
    captured = capsys.readouterr().out
    assert "after" in captured
    assert ("x:" in captured) or ("y:" in captured)


def test_diagnose_distribution_shift_flags():
    train = pd.DataFrame({"x": [0, 0, 0, 0, 0], "y": [1, 2, 3, 4, 5]})
    test = pd.DataFrame({"x": [0, 0, 0, 5, 5], "y": [1, 2, 3, 4, 5]})
    df = diagnose_distribution_shift(train, test, threshold=0.2)
    assert df.loc[df["feature"] == "x", "shifted"].iloc[0] is True


def test_verify_no_leakage_passes_and_fails():
    X_train = pd.DataFrame({"a": [0.0, 0.0, 0.0]})
    scaler_params = {"a": {"mean": 0.0}}
    # should not raise
    verify_no_leakage(X_train, scaler_params)
    # now make mean non-zero -> should raise AssertionError
    X_train2 = pd.DataFrame({"a": [1.0, 1.0, 1.0]})
    with pytest.raises(AssertionError):
        verify_no_leakage(X_train2, scaler_params)
