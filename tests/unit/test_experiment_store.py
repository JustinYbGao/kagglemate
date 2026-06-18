"""Unit tests for kagglemate.memory.experiment_store."""

from __future__ import annotations

from pathlib import Path

import pytest

from kagglemate.memory.experiment_store import ExperimentStore


def test_create_and_get_experiment(tmp_path: Path):
    store = ExperimentStore("test_comp", db_path=tmp_path / "experiments.db")
    exp_id = store.insert({
        "experiment_name": "baseline_lgbm_001",
        "model_name": "LightGBM",
        "cv_score": 0.85,
        "cv_std": 0.01,
        "metric": "accuracy",
        "task_type": "binary_classification",
        "target_column": "Survived",
        "id_column": "PassengerId",
        "cv_strategy": "StratifiedKFold",
        "features": ["Pclass", "Sex"],
        "params": {"n_estimators": 100},
    })
    record = store.get(exp_id)
    assert record is not None
    assert record["experiment_name"] == "baseline_lgbm_001"
    assert record["cv_score"] == pytest.approx(0.85)
    assert record["task_type"] == "binary_classification"
    assert record["target_column"] == "Survived"
    assert record["cv_strategy"] == "StratifiedKFold"
    assert record["features"] == ["Pclass", "Sex"]


def test_update_experiment(tmp_path: Path):
    store = ExperimentStore("test_comp", db_path=tmp_path / "experiments.db")
    exp_id = store.insert({"experiment_name": "test"})
    ok = store.update_lb(exp_id, 0.78)
    assert ok
    record = store.get(exp_id)
    assert record["lb_score"] == pytest.approx(0.78)


def test_record_artifact_paths(tmp_path: Path):
    store = ExperimentStore("test_comp", db_path=tmp_path / "experiments.db")
    exp_id = store.insert({
        "experiment_name": "test",
        "oof_path": str(tmp_path / "oof.csv"),
        "fold_scores_path": str(tmp_path / "folds.json"),
        "config_path": str(tmp_path / "config.json"),
        "strategy_validation_report_path": str(tmp_path / "svr.json"),
        "submission_validation_report_path": str(tmp_path / "subval.json"),
        "benchmark_result_path": str(tmp_path / "bench.json"),
        "script_hash": "abc123",
    })
    record = store.get(exp_id)
    assert record["oof_path"].endswith("oof.csv")
    assert record["strategy_validation_report_path"].endswith("svr.json")
    assert record["script_hash"] == "abc123"


def test_record_failed_run(tmp_path: Path):
    store = ExperimentStore("test_comp", db_path=tmp_path / "experiments.db")
    exp_id = store.insert({
        "experiment_name": "failed",
        "status": "failed",
        "error_message": "Timeout",
    })
    record = store.get(exp_id)
    assert record["status"] == "failed"
    assert record["error_message"] == "Timeout"


def test_migration_adds_new_columns(tmp_path: Path):
    """Old DBs are upgraded idempotently via MIGRATIONS."""
    db_path = tmp_path / "old_experiments.db"
    # Simulate an old schema by creating the table manually without new columns
    import sqlite3
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE experiments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            experiment_name TEXT NOT NULL,
            competition_slug TEXT NOT NULL,
            model_name TEXT NOT NULL DEFAULT 'Unknown',
            cv_score REAL,
            cv_std REAL,
            lb_score REAL,
            metric TEXT DEFAULT 'unknown',
            cv_folds INTEGER DEFAULT 5,
            features TEXT,
            params TEXT,
            feature_importance TEXT,
            fold_scores TEXT,
            oof_path TEXT,
            fold_scores_path TEXT,
            config_path TEXT,
            runtime_seconds REAL,
            script_hash TEXT,
            submission_hash TEXT,
            submission_path TEXT,
            script_path TEXT,
            report_path TEXT,
            notes TEXT,
            status TEXT DEFAULT 'completed',
            error_message TEXT,
            created_at TEXT DEFAULT (datetime('now')),
            lb_updated_at TEXT
        )
    """)
    conn.commit()
    conn.close()

    store = ExperimentStore("test_comp", db_path=db_path)
    exp_id = store.insert({
        "experiment_name": "migrated",
        "task_type": "regression",
        "cv_strategy": "KFold",
    })
    record = store.get(exp_id)
    assert record["task_type"] == "regression"
    assert record["cv_strategy"] == "KFold"


def test_get_best(tmp_path: Path):
    store = ExperimentStore("test_comp", db_path=tmp_path / "experiments.db")
    store.insert({"experiment_name": "low", "cv_score": 0.5, "status": "completed"})
    store.insert({"experiment_name": "high", "cv_score": 0.9, "status": "completed"})
    best = store.get_best()
    assert best is not None
    assert best["experiment_name"] == "high"
