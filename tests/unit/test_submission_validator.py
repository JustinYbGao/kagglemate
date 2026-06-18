"""Unit tests for kagglemate.tools.submission_validator."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from kagglemate.tools.submission_validator import validate


def test_valid_submission(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Survived": [0, 1, 0, 1, 0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert result.is_valid


def test_wrong_columns(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Prediction": [0, 1, 0, 1, 0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert not result.is_valid
    assert any(c.check == "columns_match_sample" and not c.passed for c in result.checks)


def test_wrong_row_count(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8],
        "Survived": [0, 1, 0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert not result.is_valid
    assert any(c.check == "row_count_matches" and not c.passed for c in result.checks)


def test_wrong_id_order(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [10, 9, 8, 7, 6],
        "Survived": [0, 1, 0, 1, 0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert not result.is_valid
    assert any(c.check == "row_order_matches" and not c.passed for c in result.checks)


def test_nan_predictions(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Survived": [0, 1, None, 1, 0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert not result.is_valid
    assert any(c.check == "no_nan" and not c.passed for c in result.checks)


def test_inf_predictions(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Survived": [0.0, 1.0, float("inf"), 1.0, 0.0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert not result.is_valid
    assert any(c.check == "no_inf" and not c.passed for c in result.checks)


def test_duplicate_ids(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 8, 10],
        "Survived": [0, 1, 0, 1, 0],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="accuracy")
    assert not result.is_valid
    assert any(c.check == "no_duplicate_ids" and not c.passed for c in result.checks)


def test_probability_out_of_range(sample_submission_dir: Path):
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "Survived": [0.5, 1.2, -0.1, 0.8, 0.3],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="auc")
    assert not result.is_valid
    assert any(c.check == "probability_range" and not c.passed for c in result.checks)


def test_multiclass_probability_sum(sample_submission_dir: Path):
    # Create a multiclass sample submission
    sample_path = sample_submission_dir / "sample_submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "A": [0.3, 0.3, 0.3, 0.3, 0.3],
        "B": [0.3, 0.3, 0.3, 0.3, 0.3],
        "C": [0.3, 0.3, 0.3, 0.3, 0.3],
    }).to_csv(sample_path, index=False)
    sub_path = sample_submission_dir / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8, 9, 10],
        "A": [0.3, 0.3, 0.3, 0.3, 0.3],
        "B": [0.3, 0.3, 0.3, 0.3, 0.3],
        "C": [0.3, 0.3, 0.3, 0.3, 0.3],
    }).to_csv(sub_path, index=False)
    result = validate(sub_path, sample_submission_dir, metric="logloss")
    assert not result.is_valid
    assert any(c.check == "probability_row_sum" and not c.passed for c in result.checks)


def test_regression_outlier_warning(tmp_path: Path, sample_train_for_submission: pd.DataFrame):
    sample_train_for_submission.to_csv(tmp_path / "train.csv", index=False)
    pd.DataFrame({
        "PassengerId": [6, 7, 8],
        "Survived": [0, 1, 0],
    }).to_csv(tmp_path / "sample_submission.csv", index=False)
    pd.DataFrame({
        "PassengerId": [6, 7, 8],
    }).to_csv(tmp_path / "test.csv", index=False)

    sub_path = tmp_path / "submission.csv"
    pd.DataFrame({
        "PassengerId": [6, 7, 8],
        "Survived": [0.0, 1.0, 999999.0],
    }).to_csv(sub_path, index=False)

    result = validate(sub_path, tmp_path, metric="rmse")
    assert result.is_valid  # warnings only
    assert any(c.check == "regression_outliers" and not c.passed for c in result.checks)
    assert any("outside training target range" in w for w in result.warnings)
