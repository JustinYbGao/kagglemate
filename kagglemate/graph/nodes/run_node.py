"""Run Node — executes a training script and records results.

1. Runs `python <script_path>` as a subprocess
2. Captures stdout and stderr
3. Parses the "=== RESULTS ===" JSON block from stdout
4. Validates the generated submission.csv
5. Saves the experiment to experiments.db
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
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

    _log(f"Running: {script_path}")

    # ── Execute script ──
    try:
        result = subprocess.run(
            [sys.executable, script_path],
            capture_output=True,
            text=True,
            timeout=config.SCRIPT_TIMEOUT_SECONDS,
            cwd=Path(script_path).parent.parent,  # competition root
        )
    except subprocess.TimeoutExpired:
        return _failed(exp, f"Script timed out after {config.SCRIPT_TIMEOUT_SECONDS}s")
    except Exception as e:
        return _failed(exp, f"Subprocess error: {e}")

    stdout = result.stdout
    stderr = result.stderr

    if stderr:
        # Not all stderr is bad (warnings etc.) — only flag if non-zero exit
        if result.returncode != 0:
            _log(f"Script failed (exit {result.returncode})")
            tail = stderr.strip().split("\n")[-30:]  # last 30 lines
            return _failed(exp, "\n".join(tail))

    # ── Parse structured output ──
    parsed = _parse_results(stdout)

    if parsed is None:
        return _failed(
            exp,
            f"No RESULTS block found in output.\n\n"
            f"stdout (last 500 chars):\n{stdout[-500:]}\n\n"
            f"stderr (last 500 chars):\n{stderr[-500:] if stderr else '(none)'}",
        )

    cv_score = parsed.get("cv_score", 0.0)
    cv_std = parsed.get("cv_std", 0.0)
    submission_path = parsed.get("submission_path", "")
    metric = parsed.get("metric", state.get("evaluation_metric", "unknown"))
    features = parsed.get("features", [])
    fold_scores = parsed.get("fold_scores", [])
    feature_importance = parsed.get("feature_importance", [])

    _log(f"CV Score: {cv_score:.5f} ± {cv_std:.5f} ({metric})")

    # ── Validate submission file ──
    validation_ok = True
    validation_errors: list[str] = []
    if submission_path and Path(submission_path).exists():
        try:
            from kagglemate.tools.submission_validator import validate
            vr = validate(submission_path, state.get("data_dir", ""))
            if not vr.is_valid:
                validation_ok = False
                validation_errors = [e.get("detail", str(e)) for e in vr.checks if not e.get("passed")]
                for err in validation_errors:
                    _log(f"Validation warning: {err}")
        except Exception as e:
            _log(f"Validation skipped: {e}")
    else:
        _log(f"Warning: submission file not found at {submission_path}")

    # ── Save to experiment database ──
    store = ExperimentStore(state["competition_slug"])
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
        "submission_path": submission_path,
        "script_path": script_path,
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
    match = re.search(r"=== RESULTS ===\s*\n(.*?)(?:\n\s*$|$)", stdout, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _failed(exp: dict, error_msg: str) -> dict:
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
