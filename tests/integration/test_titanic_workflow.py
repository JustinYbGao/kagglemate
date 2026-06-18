"""Integration test for the full Titanic workflow.

This test does NOT submit to Kaggle. It verifies that the KaggleMate pipeline can:
1. Load competition config
2. Generate data profile
3. Generate CV plan and cv_config.json
4. Generate a baseline script using heuristic strategy
5. Validate a dummy submission passes the submission validator
6. Record an experiment to the experiment store

If local Titanic data is unavailable, the test is skipped gracefully.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from kagglemate.config import config
from kagglemate.cv_strategy import generate_cv_plan
from kagglemate.graph.nodes.baseline_node import _render_script, _write_experiment_config
from kagglemate.graph.state import KaggleAgentState
from kagglemate.memory.experiment_store import ExperimentStore
from kagglemate.strategy_validator import heuristic_strategy, validate_and_fix
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


def _has_titanic_data() -> bool:
    """Check if required Titanic CSVs are present."""
    data_dir = config.COMPETITIONS_DIR / "titanic" / "data" / "raw"
    required = ["train.csv", "test.csv", "sample_submission.csv"]
    return all((data_dir / name).exists() for name in required)


@pytest.mark.skipif(not _has_titanic_data(), reason="Titanic data not available locally")
def test_titanic_workflow():
    comp = _load_titanic_config()
    if comp is None:
        pytest.skip("Titanic config not found")

    data_dir = config.COMPETITIONS_DIR / "titanic" / "data" / "raw"
    with tempfile.TemporaryDirectory() as tmp:
        run_dir = Path(tmp)

        # 1. Data profile
        profiler = DataProfiler(data_dir)
        profile = profiler.run()
        assert profile.get("target_col") == comp["target_column"]
        assert profile.get("id_col") == comp["id_column"]

        # 2. CV plan
        cv_plan = generate_cv_plan(
            profile,
            {"slug": comp["slug"], "type": comp["task_type"]},
            comp["metric"],
            comp["target_column"],
            run_dir,
        )
        assert cv_plan["strategy"] == "StratifiedKFold"
        assert (run_dir / "cv_plan.md").exists()
        assert (run_dir / "cv_config.json").exists()

        # 3. Build state and generate heuristic strategy
        state = KaggleAgentState(
            competition_slug=comp["slug"],
            competition_type=comp["task_type"],
            evaluation_metric=comp["metric"],
            data_dir=str(data_dir),
            report_dir=str(run_dir),
            submission_dir=str(run_dir),
            script_dir=str(run_dir),
        )
        strategy = heuristic_strategy(profile)
        val_result = validate_and_fix(strategy, profile)
        assert val_result.valid

        # 4. Generate baseline script
        config_path = _write_experiment_config(state, profile, val_result.strategy, cv_plan, val_result)
        script_path = _render_script(state, profile, val_result.strategy, cv_plan, config_path)
        assert script_path.exists()
        assert "from sklearn.model_selection import StratifiedKFold" in script_path.read_text()

        # 5. Validate a dummy valid submission
        sample_path = data_dir / "sample_submission.csv"
        assert sample_path.exists()
        vr = validate(sample_path, data_dir, metric=comp["metric"])
        assert vr.is_valid, f"Sample submission should be valid: {vr.errors}"

        # 6. Record experiment to temporary DB
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
            "strategy_validation_report_path": str(run_dir / "strategy_validation_report.json"),
        })
        record = store.get(exp_id)
        assert record is not None
        assert record["task_type"] == comp["task_type"]
        assert record["cv_strategy"] == "StratifiedKFold"
