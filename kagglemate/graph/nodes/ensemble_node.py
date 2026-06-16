"""Ensemble Node — blends multiple experiment submissions.

Methods:
- simple_average: arithmetic mean of all predictions
- weighted_average: weighted by CV score (higher CV → more weight)
- rank_average: rank each prediction, then average ranks

Used when you have 2+ experiments with submission files and want
to squeeze out extra performance from model diversity.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

import numpy as np
import pandas as pd

from kagglemate.graph.state import KaggleAgentState
from kagglemate.memory.experiment_store import ExperimentStore
from kagglemate.tools.submission_validator import validate


def run(state: KaggleAgentState) -> dict:
    """Blend multiple experiment submission files.

    Reads `ensemble_exp_ids` and `ensemble_method` from state.
    """
    slug = state["competition_slug"]
    exp_ids = state.get("ensemble_exp_ids", [])
    method = state.get("ensemble_method", "simple_average")

    if not exp_ids or len(exp_ids) < 2:
        return {
            "errors": ["Need at least 2 experiment IDs to ensemble. Use --ids 1,2,3"],
            "current_phase": "build",
        }

    store = ExperimentStore(slug)
    exps = store.compare(exp_ids)

    if len(exps) < 2:
        return {
            "errors": [f"Found only {len(exps)} of {len(exp_ids)} experiments."],
            "current_phase": "build",
        }

    _log(f"Blending {len(exps)} experiments with method: {method}")

    # ── Load submission files ──
    submissions = []
    weights = []
    for exp in exps:
        sub_path = exp.get("submission_path", "")
        if not sub_path or not Path(sub_path).exists():
            _log(f"Skipping experiment #{exp['id']}: no submission file at {sub_path}")
            continue
        df = pd.read_csv(sub_path)
        submissions.append(df)
        # Weight = CV score (higher is better for most metrics)
        cv = exp.get("cv_score") or 0.0
        weights.append(max(cv, 0.001))  # ensure positive

    if len(submissions) < 2:
        return {
            "errors": [f"Only {len(submissions)} valid submission files found. Need 2+."],
            "current_phase": "build",
        }

    _log(f"Loaded {len(submissions)} submission files")

    # ── Validate consistency ──
    first = submissions[0]
    id_col = first.columns[0]
    pred_col = first.columns[-1] if len(first.columns) > 1 else first.columns[0]

    for i, s in enumerate(submissions[1:], 1):
        if list(s.columns) != list(first.columns):
            return {
                "errors": [f"Column mismatch: submission 0 has {list(first.columns)}, "
                           f"submission {i} has {list(s.columns)}"],
                "current_phase": "build",
            }
        if len(s) != len(first):
            return {
                "errors": [f"Row count mismatch: submission 0 has {len(first)} rows, "
                           f"submission {i} has {len(s)} rows"],
                "current_phase": "build",
            }

    # ── Blend predictions ──
    ids = first[id_col].values
    preds = np.array([s[pred_col].values for s in submissions])

    if method == "simple_average":
        blended = np.mean(preds, axis=0)
        method_label = "Simple Average / 简单平均"

    elif method == "weighted_average":
        w = np.array(weights)
        w = w / w.sum()
        blended = np.average(preds, axis=0, weights=w)
        method_label = f"Weighted Average / 加权平均 (weights: {[f'{x:.3f}' for x in w]})"

    elif method == "rank_average":
        # Rank each prediction vector, then average ranks
        ranked = np.zeros_like(preds)
        for i in range(preds.shape[0]):
            from scipy.stats import rankdata
            ranked[i] = rankdata(preds[i])
        blended = np.mean(ranked, axis=0)
        # Normalize back to [0, 1] range
        blended = (blended - blended.min()) / (blended.max() - blended.min() + 1e-8)
        method_label = "Rank Average / 排序平均"

    else:
        return {
            "errors": [f"Unknown ensemble method: {method}. "
                       f"Use: simple_average, weighted_average, rank_average"],
            "current_phase": "build",
        }

    # ── Save blended submission ──
    sub_dir = state.get("submission_dir", "")
    if not sub_dir:
        sub_dir = str(Path(f"competitions/{slug}/submissions"))
    sub_path = Path(sub_dir) / f"ensemble_{method}_{len(submissions)}models.csv"
    sub_path.parent.mkdir(parents=True, exist_ok=True)

    blended_df = pd.DataFrame({id_col: ids, pred_col: blended})
    blended_df.to_csv(sub_path, index=False)

    _log(f"Blended submission saved → {sub_path}")

    # ── Validate ──
    data_dir = state.get("data_dir", "")
    vr = validate(str(sub_path), data_dir) if data_dir else None

    # ── Build summary ──
    model_summary = "\n".join(
        f"  #{e['id']}: {e.get('experiment_name', '?')} "
        f"(CV={e.get('cv_score', 'N/A')}, model={e.get('model_name', '?')})"
        for e in exps
    )

    _log(f"\nBlended from:\n{model_summary}")
    _log(f"Method: {method_label}")

    experiment = {
        "name": f"ensemble_{method}_{len(submissions)}models",
        "model": f"Ensemble({method})",
        "cv_score": 0.0,  # No CV for ensemble (would need OOF preds)
        "lb_score": None,
        "features": [],
        "submission_path": str(sub_path),
        "status": "completed",
        "notes": f"Blended submissions from experiments: {exp_ids}\n{model_summary}",
    }

    # Save to DB
    exp_id = store.insert(experiment)

    return {
        "current_experiment": {**experiment, "id": exp_id},
        "all_experiments": [{**experiment, "id": exp_id}],
        "submission_file": str(sub_path),
        "current_phase": "build",
    }


def _log(msg: str):
    print(f"  [ensemble] {msg}")
