"""Unit tests for kagglemate.cv_strategy."""

from __future__ import annotations

from pathlib import Path

import pytest

from kagglemate.cv_strategy import generate_cv_plan


def _make_profile(task_type: str = "binary_classification", n_unique_target: int = 2, date_col: bool = False, group_col: bool = False) -> dict:
    profile = {
        "train_rows": 100,
        "test_rows": 50,
        "columns": ["id", "feat1", "feat2", "target"],
        "target_col": "target",
        "id_col": "id",
        "numerical_cols": ["id", "feat1", "feat2"],
        "categorical_cols": [],
        "datetime_cols": [],
        "column_details": [
            {"name": "id", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 100},
            {"name": "feat1", "dtype": "float64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 90},
            {"name": "feat2", "dtype": "float64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 80},
            {"name": "target", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": n_unique_target},
        ],
    }
    if date_col:
        profile["columns"].append("event_date")
        profile["numerical_cols"].append("event_date")
        profile["column_details"].append(
            {"name": "event_date", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 100}
        )
    if group_col:
        profile["columns"].append("group_id")
        profile["numerical_cols"].append("group_id")
        profile["column_details"].append(
            {"name": "group_id", "dtype": "int64", "n_missing": 0, "missing_pct": 0.0, "n_unique": 10}
        )
    return profile


def test_binary_classification_uses_stratified_kfold(tmp_path: Path):
    profile = _make_profile(task_type="binary_classification", n_unique_target=2)
    plan = generate_cv_plan(profile, {"slug": "test", "type": "binary_classification"}, "accuracy", "target", tmp_path)
    assert plan["strategy"] == "StratifiedKFold"
    assert plan["cv_config"]["splitter"] == "StratifiedKFold"
    assert (tmp_path / "cv_config.json").exists()
    assert (tmp_path / "CV_PLAN.md").exists()


def test_regression_uses_kfold(tmp_path: Path):
    profile = _make_profile(task_type="regression", n_unique_target=100)
    plan = generate_cv_plan(profile, {"slug": "test", "type": "regression"}, "rmse", "target", tmp_path)
    assert plan["strategy"] == "KFold"
    assert plan["cv_config"]["splitter"] == "KFold"


def test_time_series_uses_time_series_split(tmp_path: Path):
    profile = _make_profile(task_type="time_series", date_col=True)
    plan = generate_cv_plan(profile, {"slug": "test", "type": "time_series"}, "rmse", "target", tmp_path)
    assert plan["strategy"] == "TimeSeriesSplit"
    assert plan["cv_config"]["time_column"] == "event_date"


def test_group_column_uses_group_kfold(tmp_path: Path):
    profile = _make_profile(task_type="binary_classification", group_col=True)
    plan = generate_cv_plan(profile, {"slug": "test", "type": "binary_classification"}, "accuracy", "target", tmp_path)
    assert plan["strategy"] == "GroupKFold"
    assert plan["cv_config"]["group_column"] == "group_id"


def test_unknown_task_falls_back_to_kfold(tmp_path: Path):
    profile = _make_profile(task_type="unknown", n_unique_target=100)
    plan = generate_cv_plan(profile, {"slug": "test", "type": "unknown"}, "rmse", "target", tmp_path)
    assert plan["strategy"] == "KFold"
    # KFold is the expected conservative choice for unknown regression-like tasks
    assert plan["cv_config"]["splitter"] == "KFold"
