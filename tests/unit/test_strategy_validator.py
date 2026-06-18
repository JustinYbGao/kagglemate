"""Unit tests for kagglemate.strategy_validator."""

from __future__ import annotations

import pandas as pd
import pytest

from kagglemate.strategy_validator import validate_and_fix, heuristic_strategy


def test_target_column_removed_from_features(sample_profile, sample_train_df, sample_test_df):
    strategy = {
        "src_feature_cols": ["Pclass", "Sex", "Survived"],
        "fe_feature_cols": [],
        "numerical_cols": ["Pclass"],
        "categorical_cols": ["Sex"],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {"n_estimators": 100},
    }
    result = validate_and_fix(strategy, sample_profile, sample_train_df, sample_test_df)
    assert "Survived" not in result.strategy["feature_cols"]
    assert any(issue["type"] == "target_leakage" for issue in result.issues)


def test_id_column_removed_from_features(sample_profile, sample_train_df, sample_test_df):
    strategy = {
        "src_feature_cols": ["PassengerId", "Pclass", "Sex"],
        "fe_feature_cols": [],
        "numerical_cols": ["PassengerId", "Pclass"],
        "categorical_cols": ["Sex"],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {},
    }
    result = validate_and_fix(strategy, sample_profile, sample_train_df, sample_test_df)
    assert "PassengerId" not in result.strategy["feature_cols"]


def test_nonexistent_feature_column_removed(sample_profile, sample_train_df, sample_test_df):
    strategy = {
        "src_feature_cols": ["Pclass", "NonExistent"],
        "fe_feature_cols": [],
        "numerical_cols": ["Pclass"],
        "categorical_cols": [],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {},
    }
    result = validate_and_fix(strategy, sample_profile, sample_train_df, sample_test_df)
    assert "NonExistent" not in result.strategy["feature_cols"]
    assert any(issue["type"] == "nonexistent_column" for issue in result.issues)


def test_duplicate_columns_deduplicated(sample_profile, sample_train_df, sample_test_df):
    strategy = {
        "src_feature_cols": ["Pclass", "Pclass", "Sex"],
        "fe_feature_cols": [],
        "numerical_cols": ["Pclass"],
        "categorical_cols": ["Sex"],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {},
    }
    result = validate_and_fix(strategy, sample_profile, sample_train_df, sample_test_df)
    assert result.strategy["feature_cols"].count("Pclass") == 1
    assert any(issue["type"] == "duplicate_column" for issue in result.issues)


def test_high_cardinality_categorical_warning():
    # Add a high-cardinality categorical column
    train_df = pd.DataFrame({
        "PassengerId": [1, 2, 3, 4, 5],
        "Pclass": [1, 2, 3, 1, 2],
        "Sex": ["male", "female", "female", "male", "female"],
        "Age": [22.0, 38.0, None, 35.0, 28.0],
        "Fare": [7.25, 71.28, 8.05, 53.1, 12.0],
        "Survived": [0, 1, 1, 0, 1],
        "Ticket": ["A", "B", "C", "D", "E"],
    })
    test_df = pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Pclass": [3, 1, 2, 3, 1],
        "Sex": ["male", "male", "female", "female", "male"],
        "Age": [25.0, 40.0, 18.0, None, 30.0],
        "Fare": [9.0, 65.0, 10.0, 7.0, 25.0],
        "Ticket": ["F", "G", "H", "I", "J"],
    })
    profile = {
        "train_rows": len(train_df),
        "test_rows": len(test_df),
        "columns": list(train_df.columns),
        "target_col": "Survived",
        "id_col": "PassengerId",
        "numerical_cols": ["PassengerId", "Pclass", "Age", "Fare"],
        "categorical_cols": ["Sex", "Ticket"],
        "datetime_cols": [],
        "missing_values": {},
        "column_details": [
            {"name": "PassengerId", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 5},
            {"name": "Pclass", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 3},
            {"name": "Sex", "dtype": "object", "n_missing": 0, "missing_pct": 0.0, "n_unique": 2},
            {"name": "Age", "dtype": "float64", "n_missing": 1, "missing_pct": 20.0, "n_unique": 4},
            {"name": "Fare", "dtype": "float64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 5},
            {"name": "Survived", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 2},
            {"name": "Ticket", "dtype": "object", "n_missing": 0, "missing_pct": 0.0, "n_unique": 100},
        ],
    }
    strategy = {
        "src_feature_cols": ["Pclass", "Sex", "Ticket"],
        "fe_feature_cols": [],
        "numerical_cols": ["Pclass"],
        "categorical_cols": ["Sex", "Ticket"],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {},
    }
    result = validate_and_fix(strategy, profile, train_df, test_df)
    assert any("Ticket" in w and "high-cardinality" in w.lower() for w in result.warnings)
    assert any(issue["type"] == "high_cardinality" for issue in result.issues)


def test_train_test_schema_mismatch_warning(sample_profile, sample_train_df):
    test_df_bad = pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Pclass": [3, 1, 2, 3, 1],
        # Missing Sex and Fare columns
    })
    strategy = {
        "src_feature_cols": ["Pclass", "Sex", "Fare"],
        "fe_feature_cols": [],
        "numerical_cols": ["Pclass", "Fare"],
        "categorical_cols": ["Sex"],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {},
    }
    result = validate_and_fix(strategy, sample_profile, sample_train_df, test_df_bad)
    assert any("schema mismatch" in w for w in result.warnings)


def test_fallback_to_heuristic_when_no_features(sample_profile):
    strategy = {
        "src_feature_cols": ["NonExistent1", "NonExistent2"],
        "fe_feature_cols": [],
        "numerical_cols": [],
        "categorical_cols": [],
        "feature_engineering": "",
        "model_name": "LightGBM",
        "model_params": {},
    }
    result = validate_and_fix(strategy, sample_profile)
    assert result.valid
    assert len(result.strategy["feature_cols"]) > 0
    assert result.strategy["model_name"] == "LightGBM"


def test_heuristic_strategy_returns_all_non_target_id_columns(sample_profile):
    result = heuristic_strategy(sample_profile)
    assert "Survived" not in result["feature_cols"]
    assert "PassengerId" not in result["feature_cols"]
    assert "Pclass" in result["feature_cols"]
    assert "Sex" in result["feature_cols"]
