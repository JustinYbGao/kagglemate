"""Shared pytest fixtures for KaggleMate tests."""

from __future__ import annotations

import tempfile
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def tmp_path_fixture(tmp_path: Path) -> Path:
    """Pytest-native temporary path."""
    return tmp_path


@pytest.fixture
def sample_train_df() -> pd.DataFrame:
    """A small synthetic training DataFrame."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Pclass": [1, 2, 3, 1, 2],
        "Sex": ["male", "female", "female", "male", "female"],
        "Age": [22.0, 38.0, None, 35.0, 28.0],
        "Fare": [7.25, 71.28, 8.05, 53.1, 12.0],
        "Survived": [0, 1, 1, 0, 1],
    })


@pytest.fixture
def sample_test_df() -> pd.DataFrame:
    """A small synthetic test DataFrame matching train schema."""
    return pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Pclass": [3, 1, 2, 3, 1],
        "Sex": ["male", "male", "female", "female", "male"],
        "Age": [25.0, 40.0, 18.0, None, 30.0],
        "Fare": [9.0, 65.0, 10.0, 7.0, 25.0],
    })


@pytest.fixture
def sample_profile(sample_train_df: pd.DataFrame, sample_test_df: pd.DataFrame) -> dict:
    """A DataProfile-like dict for tests."""
    numerical = ["PassengerId", "Pclass", "Age", "Fare"]
    categorical = ["Sex"]
    return {
        "train_rows": len(sample_train_df),
        "test_rows": len(sample_test_df),
        "columns": list(sample_train_df.columns),
        "target_col": "Survived",
        "id_col": "PassengerId",
        "numerical_cols": numerical,
        "categorical_cols": categorical,
        "datetime_cols": [],
        "missing_values": {"Age": 20.0},
        "target_distribution": "0: 2, 1: 3",
        "submission_cols": ["PassengerId", "Survived"],
        "submission_rows": len(sample_test_df),
        "column_details": [
            {"name": "PassengerId", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 5},
            {"name": "Pclass", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 3},
            {"name": "Sex", "dtype": "object", "n_missing": 0, "missing_pct": 0.0, "n_unique": 2},
            {"name": "Age", "dtype": "float64", "n_missing": 1, "missing_pct": 20.0, "n_unique": 4},
            {"name": "Fare", "dtype": "float64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 5},
            {"name": "Survived", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 2},
        ],
    }


@pytest.fixture
def sample_submission_valid() -> pd.DataFrame:
    """A valid submission DataFrame for binary classification."""
    return pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Survived": [0, 1, 0, 1, 0],
    })


@pytest.fixture
def sample_submission_dir(tmp_path: Path, sample_submission_valid: pd.DataFrame) -> Path:
    """Temporary directory with sample_submission.csv and test.csv."""
    sample_submission_valid.to_csv(tmp_path / "sample_submission.csv", index=False)
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Pclass": [3, 1, 2, 3, 1],
    }).to_csv(tmp_path / "test.csv", index=False)
    return tmp_path


@pytest.fixture
def sample_train_for_submission() -> pd.DataFrame:
    """Training data used for submission validator outlier tests."""
    return pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Survived": [0, 1, 0, 1, 0],
    })
