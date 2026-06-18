"""Run Node — executes a training script and records results.

1. Runs `python <script_path>` as a subprocess
2. Captures stdout and stderr, persists run_log.txt
3. Parses the "=== RESULTS ===" JSON block from stdout
4. Persists fold_scores.json and records artifact paths
5. Computes script hash and runtime
6. Validates the generated submission.csv
7. Saves the experiment to experiments.db
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
import time
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState
from kagglemate.config import config
from kagglemate.memory.experiment_store import ExperimentStore


def run(state: KaggleAgentState) -> dict:
    """Execute the most recently generated training script.

    Returns state updates with experiment results (cv_score, submission_path, etc.).
    """
    exp = state.get("current_experiment") or {}
    script_path = exp.get("script_path", "")

    if not script_path or not Path(script_path).exists():
        return {
            "errors": [f"Script not found: {script_path}. Run 'baseline' first."],
            "current_phase": "run",
        }

    script_path_obj = Path(script_path)
    competition_slug = state.get("competition_slug") or exp.get("competition_slug", "unknown")
    submission_dir = Path(state.get("submission_dir", ""))
    submission_dir.mkdir(parents=True, exist_ok=True)

    # Compute script hash before execution
    script_hash = hashlib.sha256(script_path_obj.read_bytes()).hexdigest()

    _log(f"Running: {script_path}")
    t0 = time.time()

    # ── Execute script ──
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=config.SCRIPT_TIMEOUT_SECONDS,
            cwd=script_path_obj.parent.parent,  # competition root
        )
    except subprocess.TimeoutExpired:
        return _failed(exp, f"Script timed out after {config.SCRIPT_TIMEOUT_SECONDS}s")
    except Exception as e:
        return _failed(exp, f"Subprocess error: {e}")

    runtime_seconds = time.time() - t0
    stdout = result.stdout
    stderr = result.stderr

    # ── Persist run log ──
    run_log_path = submission_dir / "run_log.txt"
    run_log_path.write_text(
        f"=== STDOUT ===\n{stdout}\n\n=== STDERR ===\n{stderr}",
        encoding="utf-8",
    )

    if stderr:
        # Not all stderr is bad (warnings etc.) — only flag if non-zero exit
        if result.returncode != 0:
            _log(f"Script failed (exit {result.returncode})")
            tail = stderr.strip().split("\n")[-30:]  # last 30 lines
            return _failed(exp, "\n".join(tail), runtime_seconds=runtime_seconds, script_hash=script_hash)

    # ── Parse structured output ──
    parsed = _parse_results(stdout)

    if parsed is None:
        return _failed(
            exp,
            f"No RESULTS block found in output.\n\n"
            f"stdout (last 500 chars):\n{stdout[-500:]}\n\n"
            f"stderr (last 500 chars):\n{stderr[-500:] if stderr else '(none)'}",
            runtime_seconds=runtime_seconds,
            script_hash=script_hash,
        )

    cv_score = parsed.get("cv_score", 0.0)
    cv_std = parsed.get("cv_std", 0.0)
    submission_path = parsed.get("submission_path", "")
    oof_path = parsed.get("oof_path", "")
    config_path = parsed.get("config_path", exp.get("config_path", ""))
    metric = parsed.get("metric", state.get("evaluation_metric", "unknown"))
    features = parsed.get("features", [])
    fold_scores = parsed.get("fold_scores", [])
    feature_importance = parsed.get("feature_importance", [])

    _log(f"CV Score: {cv_score:.5f} ± {cv_std:.5f} ({metric})")

    # ── Persist fold scores ──
    fold_scores_path = submission_dir / "fold_scores.json"
    fold_scores_path.write_text(
        json.dumps({
            "metric": metric,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "fold_scores": fold_scores,
            "generated_at": datetime.now(timezone.utc).isoformat(),
        }, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── Validate submission file ──
    validation_ok = True
    validation_errors: list[str] = []
    validation_report_path = ""
    if submission_path and Path(submission_path).exists():
        try:
            from kagglemate.tools.submission_validator import validate, save_validation_report
            vr = validate(submission_path, state.get("data_dir", ""), metric=metric, competition_slug=competition_slug)
            validation_report_path = str(Path(submission_path).parent / "submission_validation_report.json")
            save_validation_report(vr, Path(validation_report_path))
            if not vr.is_valid:
                validation_ok = False
                validation_errors = [e.get("detail", str(e)) for e in vr.checks if not e.get("passed")]
                for err in validation_errors:
                    _log(f"Validation warning: {err}")
            for w in vr.warnings:
                _log(f"Validation warning: {w}")
        except Exception as e:
            _log(f"Validation skipped: {e}")
    else:
        _log(f"Warning: submission file not found at {submission_path}")

    # ── Compute submission hash ──
    submission_hash = ""
    if submission_path and Path(submission_path).exists():
        submission_hash = hashlib.sha256(Path(submission_path).read_bytes()).hexdigest()

    # ── Save to experiment database ──
    store = ExperimentStore(competition_slug)
    exp_id = store.insert({
        "experiment_name": exp.get("name", f"baseline_{parsed.get('model', 'lgbm')}_001"),
        "model_name": parsed.get("model") or exp.get("model") or "Unknown",
        "cv_score": cv_score,
        "cv_std": cv_std,
        "metric": metric,
        "cv_folds": parsed.get("cv_folds", 5),
        "features": features,
        "params": exp.get("params", {}),
        "feature_importance": feature_importance,
        "fold_scores": fold_scores,
        "oof_path": oof_path,
        "fold_scores_path": str(fold_scores_path),
        "config_path": config_path,
        "runtime_seconds": runtime_seconds,
        "script_hash": script_hash,
        "submission_hash": submission_hash,
        "submission_path": submission_path,
        "script_path": script_path,
        "strategy_validation_report_path": exp.get("strategy_validation_report_path", ""),
        "submission_validation_report_path": validation_report_path,
        "status": "completed" if result.returncode == 0 else "failed",
    })

    _log(f"Experiment #{exp_id} saved to experiments.db")

    # ── Update best scores ──
    best = store.get_best()
    best_cv = best["cv_score"] if best else cv_score
    best_lb = best.get("lb_score") if best and best.get("lb_score") else 0.0

    return {
        "current_experiment": {
            **exp,
            "id": exp_id,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "submission_path": submission_path,
            "oof_path": oof_path,
            "fold_scores_path": str(fold_scores_path),
            "config_path": config_path,
            "submission_validation_report_path": validation_report_path,
            "runtime_seconds": runtime_seconds,
            "script_hash": script_hash,
            "submission_hash": submission_hash,
            "status": "completed",
        },
        "all_experiments": [{
            **exp,
            "id": exp_id,
            "cv_score": cv_score,
            "cv_std": cv_std,
            "submission_path": submission_path,
            "status": "completed",
        }],
        "best_cv_score": best_cv,
        "best_lb_score": best_lb,
        "current_phase": "run",
        "errors": validation_errors if validation_errors else [],
    }


def _parse_results(stdout: str) -> dict | None:
    """Extract the === RESULTS === JSON block from script output."""
    match = re.search(r"=== RESULTS ===\s*\n(\{.*?\})\s*\n", stdout, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _failed(exp: dict, error_msg: str, runtime_seconds: float | None = None, script_hash: str = "") -> dict:
    """Record a failed experiment."""
    _log(f"FAILED: {error_msg[:200]}")

    # Try to save to DB anyway (for tracking)
    if exp.get("script_path"):
        try:
            store = ExperimentStore(exp.get("competition_slug", "unknown"))
            store.insert({
                "experiment_name": exp.get("name", "baseline_001"),
                "model_name": exp.get("model", "Unknown"),
                "status": "failed",
                "error_message": error_msg[:2000],
                "script_path": exp.get("script_path", ""),
                "config_path": exp.get("config_path", ""),
                "runtime_seconds": runtime_seconds,
                "script_hash": script_hash,
            })
        except Exception:
            pass

    return {
        "current_experiment": {**exp, "status": "failed", "error_message": error_msg[:500]},
        "errors": [f"Run failed: {error_msg[:500]}"],
        "current_phase": "run",
    }


def _log(msg: str):
    print(f"  [run] {msg}")
