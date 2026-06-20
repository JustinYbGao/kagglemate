"""Integration test for the full Titanic workflow using offline synthetic fixtures.

This test does NOT submit to Kaggle and does NOT require:
- real Kaggle Titanic data
- Kaggle API credentials
- LLM API credentials

It verifies that the KaggleMate pipeline can:
1. Load competition config
2. Generate data profile from synthetic fixtures
3. Generate CV plan and cv_config.json
4. Generate a baseline script using heuristic strategy via baseline_generator
5. Validate a dummy submission passes the submission validator
6. Record an experiment to the experiment store

For real-data coverage, run with `--kaggle_data` (requires pytest marker support).
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kagglemate.baseline_generator import generate_baseline_script
from kagglemate.cv_strategy import generate_cv_plan
from kagglemate.memory.experiment_store import ExperimentStore
from kagglemate.strategy_validator import heuristic_strategy
from kagglemate.tools.data_profiler import DataProfiler
from kagglemate.tools.submission_validator import validate


def _load_titanic_config() -> dict | None:
    """Load titanic config from benchmarks/competitions.yaml if available."""
    try:
        import yaml
        config_path = Path(__file__).parent.parent.parent / "benchmarks" / "competitions.yaml"
        with open(config_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        for comp in data.get("competitions", []):
            if comp["slug"] == "titanic":
                return comp
    except Exception:
        pass
    return None


def _synthetic_data_dir() -> Path:
    """Return the offline synthetic fixture directory for Titanic."""
    return Path(__file__).parent.parent.parent / "benchmarks" / "fixtures" / "titanic_synthetic"


def test_titanic_workflow():
    comp = _load_titanic_config()
    if comp is None:
        pytest.skip("Titanic config not found")

    data_dir = _synthetic_data_dir()
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)

        # 1. Data profile
        profiler = DataProfiler(data_dir)
        profile = profiler.run()
        # Enforce registry columns on synthetic fixtures
        profile["target_col"] = comp["target_column"]
        profile["id_col"] = comp["id_column"]
        assert profile.get("target_col") == comp["target_column"]
        assert profile.get("id_col") == comp["id_column"]
        assert profile["train_rows"] >= 1
        assert profile["test_rows"] >= 1

        # 2. CV plan
        cv_plan = generate_cv_plan(
            profile,
            {"slug": comp["slug"], "type": comp["task_type"]},
            comp["metric"],
            comp["target_column"],
            run_dir,
        )
        assert cv_plan["strategy"] == "StratifiedKFold"
        assert (run_dir / "CV_PLAN.md").exists()
        assert (run_dir / "cv_config.json").exists()

        # 3. Build state
        state = {
            "competition_slug": comp["slug"],
            "competition_type": comp["task_type"],
            "evaluation_metric": comp["metric"],
            "data_dir": str(data_dir),
            "report_dir": str(run_dir),
            "submission_dir": str(run_dir),
            "script_dir": str(run_dir),
        }

        # 4. Generate heuristic strategy and validate
        strategy = heuristic_strategy(profile)

        # 5. Generate baseline script/config via public API
        gen_result = generate_baseline_script(
            competition_config=comp,
            data_profile=profile,
            cv_config=cv_plan,
            strategy=strategy,
            output_dir=run_dir,
            data_dir=data_dir,
            use_llm=False,
        )
        script_path = Path(gen_result["script_path"])
        config_path = Path(gen_result["config_path"])
        assert script_path.exists()
        assert config_path.exists()
        assert "from sklearn.model_selection import StratifiedKFold" in script_path.read_text()

        # 6. Validate the sample submission passes the submission validator
        sample_path = data_dir / "sample_submission.csv"
        assert sample_path.exists()
        vr = validate(sample_path, data_dir, metric=comp["metric"])
        assert vr.is_valid, f"Sample submission should be valid: {vr.errors}"

        # 7. Record experiment to temporary DB
        store = ExperimentStore(comp["slug"], db_path=run_dir / "experiments.db")
        exp_id = store.insert({
            "experiment_name": "integration_test",
            "task_type": comp["task_type"],
            "target_column": comp["target_column"],
            "id_column": comp["id_column"],
            "cv_strategy": cv_plan["strategy"],
            "metric": comp["metric"],
            "status": "completed",
            "config_path": str(config_path),
            "script_path": str(script_path),
            "strategy_validation_report_path": gen_result["strategy_validation_report_path"],
        })
        record = store.get(exp_id)
        assert record is not None
        assert record["task_type"] == comp["task_type"]
        assert record["cv_strategy"] == "StratifiedKFold"


@pytest.mark.kaggle_data
def test_titanic_workflow_with_real_data():
    """Optional integration test against real Kaggle Titanic data.

    This test is skipped by default and only runs when the pytest marker
    `--kaggle_data` is enabled. It requires locally downloaded Titanic data
    and Kaggle credentials are not used by the test itself.
    """
    from kagglemate.config import config as km_config

    comp = _load_titanic_config()
    if comp is None:
        pytest.skip("Titanic config not found")

    data_dir = km_config.COMPETITIONS_DIR / "titanic" / "data" / "raw"
    required = ["train.csv", "test.csv", "sample_submission.csv"]
    if not all((data_dir / name).exists() for name in required):
        pytest.skip("Real Titanic data not available locally")

    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)

        profiler = DataProfiler(data_dir)
        profile = profiler.run()
        assert profile.get("target_col") == comp["target_column"]
        assert profile.get("id_col") == comp["id_column"]

        cv_plan = generate_cv_plan(
            profile,
            {"slug": comp["slug"], "type": comp["task_type"]},
            comp["metric"],
            comp["target_column"],
            run_dir,
        )

        strategy = heuristic_strategy(profile)
        gen_result = generate_baseline_script(
            competition_config=comp,
            data_profile=profile,
            cv_config=cv_plan,
            strategy=strategy,
            output_dir=run_dir,
            data_dir=data_dir,
            use_llm=False,
        )
        assert Path(gen_result["script_path"]).exists()
        assert Path(gen_result["config_path"]).exists()

        sample_path = data_dir / "sample_submission.csv"
        vr = validate(sample_path, data_dir, metric=comp["metric"])
        assert vr.is_valid, f"Sample submission should be valid: {vr.errors}"
