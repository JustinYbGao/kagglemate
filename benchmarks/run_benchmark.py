"""Benchmark runner for KaggleMate.

Runs the full tabular competition workflow for one or more configured competitions
and produces reproducible benchmark artifacts.

By default the runner uses offline synthetic fixtures so the benchmark is
fully reproducible without Kaggle credentials, Kaggle API access, or real data.

Usage:
    python benchmarks/run_benchmark.py --competition titanic --synthetic
    python benchmarks/run_benchmark.py --all --synthetic
    python benchmarks/run_benchmark.py --all --synthetic --dry-run
    python benchmarks/run_benchmark.py --competition titanic --data-dir competitions/titanic/data/raw
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

# Make kagglemate importable when running from repo root
sys.path.insert(0, str(Path(__file__).parent.parent))

from kagglemate.baseline_generator import generate_baseline_script
from kagglemate.config import config
from kagglemate.cv_strategy import generate_cv_plan
from kagglemate.memory.experiment_store import ExperimentStore
from kagglemate.strategy_validator import heuristic_strategy
from kagglemate.tools.data_profiler import DataProfiler
from kagglemate.tools.submission_validator import validate


BENCHMARKS_DIR = Path(__file__).parent
CONFIG_PATH = BENCHMARKS_DIR / "competitions.yaml"
RESULTS_DIR = BENCHMARKS_DIR / "results"
REPORTS_DIR = Path(__file__).parent.parent / "reports"
REQUIRED_FILES = {"train.csv", "test.csv", "sample_submission.csv"}


def load_competition_config(path: Path | str = CONFIG_PATH) -> list[dict]:
    """Load the benchmark competition registry."""
    with open(path, "r", encoding="utf-8") as f:
        data = yaml.safe_load(f)
    return data.get("competitions", [])


def default_data_dir(slug: str) -> Path:
    """Default competition data location."""
    return config.COMPETITIONS_DIR / slug / "data" / "raw"


def synthetic_fixture_dir(competition: dict) -> Path | None:
    """Return the configured synthetic fixture directory, or None."""
    fixture = competition.get("synthetic_fixture")
    if not fixture:
        return None
    return BENCHMARKS_DIR.parent / fixture


def resolve_data_dir(
    competition: dict,
    *,
    synthetic: bool = False,
    data_dir: Path | str | None = None,
) -> Path | None:
    """Resolve the data directory based on CLI flags.

    Priority:
    1. --synthetic → competitions.yaml synthetic_fixture
    2. --data-dir  → user-specified directory
    3. default     → competitions/<slug>/data/raw
    """
    if synthetic:
        fixture = synthetic_fixture_dir(competition)
        if fixture and fixture.exists():
            return fixture
        return None
    if data_dir is not None:
        return Path(data_dir)
    return default_data_dir(competition["slug"])


def has_required_data(data_dir: Path) -> bool:
    """Return True if all required CSV files exist."""
    if not data_dir.exists():
        return False
    return all((data_dir / name).exists() for name in REQUIRED_FILES)


def make_run_dir(slug: str) -> Path:
    """Create a timestamped benchmark run directory."""
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
    run_dir = RESULTS_DIR / slug / timestamp
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def save_json(data: Any, path: Path) -> Path:
    """Pretty-print JSON to path and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False, default=str)
    return path


def make_benchmark_result(competition: dict) -> dict:
    """Initialize a benchmark result skeleton."""
    return {
        "competition": competition["slug"],
        "name": competition["name"],
        "task_type": competition["task_type"],
        "metric": competition["metric"],
        "target_column": competition["target_column"],
        "id_column": competition["id_column"],
        "expected_submission_columns": competition.get("expected_submission_columns", []),
        "workflow_completed": False,
        "valid_submission": False,
        "cv_score": None,
        "cv_std": None,
        "runtime_seconds": None,
        "errors": [],
        "warnings": [],
        "artifacts": {},
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def make_state(competition: dict, data_dir: Path, run_dir: Path) -> dict:
    """Construct a plain state dict for the benchmark run."""
    return {
        "competition_slug": competition["slug"],
        "competition_type": competition["task_type"],
        "evaluation_metric": competition["metric"],
        "data_dir": str(data_dir),
        "report_dir": str(run_dir),
        "submission_dir": str(run_dir),
        "script_dir": str(run_dir),
    }


def run_single(
    competition: dict,
    *,
    synthetic: bool = False,
    data_dir: Path | str | None = None,
    use_llm: bool = False,
    dry_run: bool = False,
) -> dict:
    """Run the full benchmark workflow for a single competition."""
    result = make_benchmark_result(competition)
    slug = competition["slug"]
    run_dir = make_run_dir(slug)
    result["run_dir"] = str(run_dir)

    resolved_data_dir = resolve_data_dir(competition, synthetic=synthetic, data_dir=data_dir)

    if resolved_data_dir is None or not has_required_data(resolved_data_dir):
        if synthetic:
            msg = (
                f"Synthetic fixture not configured or missing for {slug}. "
                f"Expected: {competition.get('synthetic_fixture')}"
            )
        elif data_dir is not None:
            msg = f"Required data not found in {data_dir}."
        else:
            msg = (
                f"Required data not found in {resolved_data_dir}. "
                "Please download it from Kaggle first, e.g.: "
                f"kaggle competitions download -c {slug}"
            )
        result["errors"].append(msg)
        save_json(result, run_dir / "benchmark_result.json")
        return result

    data_dir = resolved_data_dir
    t0 = time.time()

    try:
        # 1. Data profiling
        profiler = DataProfiler(data_dir)
        profile = profiler.run()
        # Enforce the registry target/id columns when using synthetic fixtures
        profile["target_col"] = competition["target_column"]
        profile["id_col"] = competition["id_column"]
        save_json(profile, run_dir / "data_profile.json")

        # 2. CV plan (saves cv_plan.md and cv_config.json to run_dir)
        cv_plan = generate_cv_plan(
            profile,
            {"slug": slug, "type": competition["task_type"]},
            competition["metric"],
            competition["target_column"],
            run_dir,
        )

        # 3. Build state
        state = make_state(competition, data_dir, run_dir)

        # 4. Feature strategy (heuristic by default; LLM only when requested)
        if use_llm:
            from kagglemate.graph.nodes.baseline_node import _get_strategy
            strategy = _get_strategy(state, profile)
        else:
            strategy = heuristic_strategy(profile)

        # 5. Validate strategy and generate script/config
        gen_result = generate_baseline_script(
            competition_config=competition,
            data_profile=profile,
            cv_config=cv_plan,
            strategy=strategy,
            output_dir=run_dir,
            data_dir=data_dir,
            use_llm=False,
        )
        config_path = Path(gen_result["config_path"])
        script_path = Path(gen_result["script_path"])
        strategy = gen_result["strategy"]
        val_report_path = Path(gen_result["strategy_validation_report_path"])

        result["warnings"].extend(strategy.get("validation_warnings", []))

        # 6. Dry-run: generate artifacts, skip execution
        if dry_run:
            result["workflow_completed"] = True
            result["valid_submission"] = True
            result["runtime_seconds"] = round(time.time() - t0, 2)
            result["warnings"].append("Dry run: script generation succeeded, execution skipped.")
            result["artifacts"] = {
                "data_profile": str(run_dir / "data_profile.json"),
                "cv_plan": str(run_dir / "CV_PLAN.md"),
                "cv_config": str(run_dir / "cv_config.json"),
                "experiment_config": str(config_path),
                "script": str(script_path),
                "strategy_validation_report": str(val_report_path),
                "benchmark_result": str(run_dir / "benchmark_result.json"),
            }
            save_json(result, run_dir / "benchmark_result.json")
            return result

        # 7. Run script
        state["current_experiment"] = {
            "name": f"benchmark_{slug}",
            "model": strategy.get("model_name", "LightGBM"),
            "script_path": str(script_path),
            "config_path": str(config_path),
            "params": strategy.get("model_params", {}),
            "status": "pending",
        }

        from kagglemate.graph.nodes import run_node
        run_result = run_node.run(state)
        exp = run_result.get("current_experiment", {})

        result["cv_score"] = exp.get("cv_score")
        result["cv_std"] = exp.get("cv_std")
        result["runtime_seconds"] = exp.get("runtime_seconds")
        result["workflow_completed"] = exp.get("status") == "completed"
        result["valid_submission"] = exp.get("status") == "completed"

        # 8. Validate submission
        submission_path = exp.get("submission_path", "")
        if submission_path and Path(submission_path).exists():
            sub_val = validate(
                submission_path,
                data_dir,
                metric=competition["metric"],
                competition_slug=slug,
            )
            save_json(sub_val.model_dump(), run_dir / "submission_validation_report.json")
            result["valid_submission"] = sub_val.is_valid and result["valid_submission"]
            result["warnings"].extend(sub_val.warnings)
            if not sub_val.is_valid:
                result["errors"].extend(sub_val.errors)
        else:
            result["valid_submission"] = False
            result["errors"].append("Submission file not generated.")

        # 9. Enrich experiment record with benchmark metadata
        exp_id = exp.get("id")
        if exp_id:
            store = ExperimentStore(slug)
            store.update_field(exp_id, "task_type", competition["task_type"])
            store.update_field(exp_id, "target_column", competition["target_column"])
            store.update_field(exp_id, "id_column", competition["id_column"])
            store.update_field(exp_id, "cv_strategy", cv_plan.get("strategy", ""))
            store.update_field(
                exp_id,
                "strategy_validation_report_path",
                str(val_report_path),
            )
            store.update_field(
                exp_id,
                "submission_validation_report_path",
                str(run_dir / "submission_validation_report.json"),
            )
            store.update_field(
                exp_id, "benchmark_result_path", str(run_dir / "benchmark_result.json")
            )

        # 10. Record artifact paths
        result["artifacts"] = {
            "data_profile": str(run_dir / "data_profile.json"),
            "cv_plan": str(run_dir / "CV_PLAN.md"),
            "cv_config": str(run_dir / "cv_config.json"),
            "experiment_config": str(config_path),
            "script": str(script_path),
            "fold_scores": exp.get("fold_scores_path"),
            "oof_pred": exp.get("oof_path"),
            "submission": exp.get("submission_path"),
            "run_log": str(Path(submission_path).parent / "run_log.txt") if submission_path else None,
            "strategy_validation_report": str(val_report_path),
            "submission_validation_report": str(run_dir / "submission_validation_report.json"),
            "benchmark_result": str(run_dir / "benchmark_result.json"),
        }

    except Exception as e:
        result["errors"].append(f"{type(e).__name__}: {e}")

    save_json(result, run_dir / "benchmark_result.json")
    return result


def main() -> int:
    parser = argparse.ArgumentParser(description="KaggleMate benchmark runner")
    parser.add_argument("--competition", type=str, help="Run a single competition slug")
    parser.add_argument("--all", action="store_true", help="Run all configured competitions")
    parser.add_argument(
        "--synthetic",
        action="store_true",
        help="Use offline synthetic fixtures instead of real Kaggle data",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        help="Path to a custom competition data directory (overrides default)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Generate scripts but do not execute")
    parser.add_argument("--use-llm", action="store_true", help="Use LLM for feature strategy (default: heuristic)")
    parser.add_argument(
        "--config", type=str, default=str(CONFIG_PATH), help="Path to competitions.yaml"
    )
    args = parser.parse_args()

    competitions = load_competition_config(args.config)

    if not args.competition and not args.all:
        parser.error("Specify --competition SLUG or --all")

    if args.competition:
        selected = [c for c in competitions if c["slug"] == args.competition]
        if not selected:
            print(f"Unknown competition: {args.competition}", file=sys.stderr)
            return 1
    else:
        selected = competitions

    data_dir = Path(args.data_dir) if args.data_dir else None

    results: list[dict] = []
    for comp in selected:
        print(f"\n{'=' * 60}")
        print(f"Benchmark: {comp['name']} ({comp['slug']})")
        print(f"{'=' * 60}")
        result = run_single(
            comp,
            synthetic=args.synthetic,
            data_dir=data_dir,
            use_llm=args.use_llm,
            dry_run=args.dry_run,
        )
        results.append(result)
        status = "✅" if result["workflow_completed"] and result["valid_submission"] else "❌"
        print(f"{status} {comp['slug']}: workflow_completed={result['workflow_completed']}, "
              f"valid_submission={result['valid_submission']}")
        if result["errors"]:
            for err in result["errors"]:
                print(f"   Error: {err}")

    print(f"\nBenchmark results written to: {RESULTS_DIR}")
    print("Run `python benchmarks/update_reports.py` to regenerate markdown reports.")
    return 0 if all(r["workflow_completed"] and r["valid_submission"] for r in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
