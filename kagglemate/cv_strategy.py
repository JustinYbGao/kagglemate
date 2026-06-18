"""CV strategy planner — selects the right cross-validation splitter for a tabular competition.

This module is intentionally deterministic and LLM-free: it reads the data profile,
competition metadata, and metric, then decides between KFold / StratifiedKFold /
GroupKFold / TimeSeriesSplit. When the signal is ambiguous it falls back to the
most conservative option and documents the risk.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from kagglemate.graph.state import DataProfile
from kagglemate.config import config


DEFAULT_SEED = 42
DEFAULT_N_FOLDS = config.DEFAULT_CV_FOLDS
MAX_STRATIFY_CLASSES = 20
GROUP_KEYWORDS = {"group", "subject", "patient", "session", "cluster", "site", "center"}
DATE_KEYWORDS = {"date", "time", "year", "month", "day", "period", "timestamp", "week"}


def generate_cv_plan(
    profile: DataProfile,
    competition_metadata: dict,
    metric: str,
    target_col: str,
    report_dir: str | Path,
) -> dict:
    """Generate a cross-validation plan and write it to CV_PLAN.md.

    Args:
        profile: DataProfile dict produced by DataProfiler.
        competition_metadata: Dict with at least "slug" and "type" keys.
        metric: Evaluation metric name (e.g. "auc", "rmse", "logloss").
        target_col: Name of the target column.
        report_dir: Directory where CV_PLAN.md will be written.

    Returns:
        A dict with keys: strategy, n_folds, shuffle, random_seed, group_col,
        date_col, cv_import, cv_setup, cv_split_args, reasoning, risk_notes.
    """
    report_dir = Path(report_dir)
    slug = competition_metadata.get("slug", "unknown")
    comp_type = (competition_metadata.get("type") or "tabular_classification").lower()
    metric = (metric or "").lower()

    train_rows = profile.get("train_rows", 0)
    columns = profile.get("columns", [])
    column_details = profile.get("column_details", [])
    is_classification = _is_classification(profile, metric)
    n_classes = _estimate_n_classes(profile, target_col)

    risk_notes: list[str] = []

    # Detect special columns
    date_col = _detect_date_col(profile)
    group_col = _detect_group_col(profile)

    # Prefer explicit datetime_cols from profiler if available
    if not date_col and profile.get("datetime_cols"):
        date_col = profile["datetime_cols"][0]

    # Decision tree
    if comp_type == "time_series" or date_col is not None:
        strategy = "TimeSeriesSplit"
        shuffle = False
        if comp_type != "time_series" and date_col:
            risk_notes.append(
                f"Detected date-like column '{date_col}'. Using TimeSeriesSplit; "
                "verify that the column is actually monotonic in time."
            )
    elif group_col is not None:
        strategy = "GroupKFold"
        shuffle = False
        if group_col == competition_metadata.get("group_col_hint"):
            risk_notes.append(f"Using user-hinted group column '{group_col}'.")
        else:
            risk_notes.append(
                f"Detected potential group column '{group_col}'. Using GroupKFold; "
                "verify that rows from the same group must stay in the same fold."
            )
    elif is_classification and n_classes is not None and n_classes <= MAX_STRATIFY_CLASSES:
        strategy = "StratifiedKFold"
        shuffle = True
    else:
        strategy = "KFold"
        shuffle = True
        if is_classification:
            risk_notes.append(
                f"Target has {n_classes} classes (> {MAX_STRATIFY_CLASSES}) or is highly imbalanced. "
                "Falling back to KFold; consider StratifiedKFold if class balance matters."
            )

    # Build sklearn code fragments
    cv_class, cv_setup, cv_split_args = _build_sklearn_cv(
        strategy=strategy,
        n_folds=DEFAULT_N_FOLDS,
        shuffle=shuffle,
        random_seed=DEFAULT_SEED,
    )

    reasoning = _build_reasoning(
        strategy=strategy,
        comp_type=comp_type,
        metric=metric,
        is_classification=is_classification,
        n_classes=n_classes,
        group_col=group_col,
        date_col=date_col,
    )

    plan = {
        "strategy": strategy,
        "n_folds": DEFAULT_N_FOLDS,
        "shuffle": shuffle,
        "random_seed": DEFAULT_SEED,
        "group_col": group_col,
        "date_col": date_col,
        "cv_import": cv_class,
        "cv_setup": cv_setup,
        "cv_split_args": cv_split_args,
        "reasoning": reasoning,
        "risk_notes": risk_notes,
        "metric": metric,
        "is_classification": is_classification,
        "n_classes": n_classes,
    }

    _write_plan_md(plan, profile, slug, report_dir)
    return plan


def _build_sklearn_cv(strategy: str, n_folds: int, shuffle: bool, random_seed: int) -> tuple[str, str, str]:
    """Return (class name, splitter init, split arguments)."""
    if strategy == "StratifiedKFold":
        return (
            "StratifiedKFold",
            f"StratifiedKFold(n_splits={n_folds}, shuffle={shuffle}, random_state={random_seed})",
            "X, y",
        )
    if strategy == "GroupKFold":
        return (
            "GroupKFold",
            f"GroupKFold(n_splits={n_folds})",
            "X, y, groups",
        )
    if strategy == "TimeSeriesSplit":
        return (
            "TimeSeriesSplit",
            f"TimeSeriesSplit(n_splits={n_folds})",
            "X",
        )
    # KFold fallback
    return (
        "KFold",
        f"KFold(n_splits={n_folds}, shuffle={shuffle}, random_state={random_seed})",
        "X, y",
    )


def _is_classification(profile: DataProfile, metric: str) -> bool:
    """Return True if the task appears to be classification."""
    metric = metric.lower()
    classification_metrics = {"auc", "roc_auc", "logloss", "accuracy", "f1", "precision", "recall", "gini"}
    regression_metrics = {"rmse", "mse", "mae", "rmsle", "r2", "mean_squared_error", "mean_absolute_error"}
    if any(m in metric for m in classification_metrics):
        return True
    if any(m in metric for m in regression_metrics):
        return False
    target = profile.get("target_col")
    if not target:
        # Default conservative assumption for tabular when metric is unknown
        return True
    n_classes = _estimate_n_classes(profile, target)
    return n_classes is not None and n_classes <= MAX_STRATIFY_CLASSES


def _estimate_n_classes(profile: DataProfile, target_col: str) -> Optional[int]:
    """Estimate number of target classes from the profile if available."""
    if not target_col:
        return None
    for cd in profile.get("column_details", []):
        if cd.get("name") == target_col:
            unique = cd.get("n_unique")
            if isinstance(unique, int) and unique >= 0:
                return unique
    return None


def _detect_group_col(profile: DataProfile) -> Optional[str]:
    """Detect a likely group column from column names.

    Conservative: only flag columns whose names explicitly suggest grouping.
    Card-based detection is too error-prone for an MVP (e.g. categorical features
    like 'Sex' or 'Ticket' on Titanic would be misclassified).
    """
    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")

    candidates: list[str] = []
    for cd in profile.get("column_details", []):
        col = cd.get("name", "")
        if col in (id_col, target_col):
            continue
        col_lower = col.lower()
        if any(kw in col_lower for kw in GROUP_KEYWORDS):
            candidates.append(col)

    return candidates[0] if candidates else None


def _detect_date_col(profile: DataProfile) -> Optional[str]:
    """Detect a likely date/time column from column names or dtypes."""
    numerical_cols = set(profile.get("numerical_cols", []))
    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")

    date_candidates: list[str] = []
    for cd in profile.get("column_details", []):
        col = cd.get("name", "")
        if col in (id_col, target_col):
            continue
        if any(kw in col.lower() for kw in DATE_KEYWORDS):
            date_candidates.append(col)
    return date_candidates[0] if date_candidates else None


def _build_reasoning(
    strategy: str,
    comp_type: str,
    metric: str,
    is_classification: bool,
    n_classes: Optional[int],
    group_col: Optional[str],
    date_col: Optional[str],
) -> str:
    parts = [f"Competition type is '{comp_type}' and metric is '{metric}'."]
    if is_classification and n_classes is not None:
        parts.append(f"Target appears to be classification with {n_classes} classes.")
    elif not is_classification:
        parts.append("Target appears to be regression.")

    if strategy == "TimeSeriesSplit":
        reason = "time series"
        if date_col:
            reason += f" (date column '{date_col}' detected)"
        parts.append(f"Selected {strategy} because {reason}.")
    elif strategy == "GroupKFold":
        parts.append(f"Selected {strategy} to avoid group leakage via column '{group_col}'.")
    elif strategy == "StratifiedKFold":
        parts.append(
            f"Selected {strategy} to preserve class distribution across {DEFAULT_N_FOLDS} folds."
        )
    else:
        parts.append(
            f"Selected {strategy} as the conservative default for this task/metric combination."
        )
    return " ".join(parts)


def _write_plan_md(plan: dict, profile: DataProfile, slug: str, report_dir: Path) -> Path:
    """Write a human-readable CV_PLAN.md."""
    report_dir.mkdir(parents=True, exist_ok=True)
    path = report_dir / "CV_PLAN.md"

    lines = [
        "# Cross-Validation Plan\n",
        f"**Competition:** `{slug}`\n",
        f"**Strategy:** `{plan['strategy']}`\n",
        f"**Folds:** {plan['n_folds']}  ",
        f"**Shuffle:** {plan['shuffle']}  ",
        f"**Random seed:** {plan['random_seed']}\n",
        "\n## Rationale\n",
        f"{plan['reasoning']}\n",
        "\n## Sklearn Setup\n",
        "```python\n",
        f"from sklearn.model_selection import {plan['cv_import']}\n",
        f"folds = {plan['cv_setup']}\n",
        f"for train_idx, val_idx in folds.split({plan['cv_split_args']}):\n",
        "    ...\n",
        "```\n",
    ]

    if plan["group_col"]:
        lines += ["\n## Group Column\n", f"`{plan['group_col']}`\n"]
    if plan["date_col"]:
        lines += ["\n## Date Column\n", f"`{plan['date_col']}`\n"]

    lines += [
        "\n## Metric Alignment\n",
        f"- Metric: `{plan['metric']}`\n",
        f"- Classification: `{plan['is_classification']}`\n",
    ]
    if plan["n_classes"] is not None:
        lines.append(f"- Estimated classes: `{plan['n_classes']}`\n")

    if plan["risk_notes"]:
        lines += ["\n## Risk Notes\n"]
        for note in plan["risk_notes"]:
            lines.append(f"- ⚠️ {note}\n")
    else:
        lines += ["\n## Risk Notes\n", "- No known risks for this strategy.\n"]

    lines += [
        "\n## Fallback\n",
        "If this strategy produces unstable CV/LB gap, fall back to `KFold(n_splits=5, "
        "shuffle=True, random_state=42)` and compare.\n",
    ]

    path.write_text("".join(lines), encoding="utf-8")
    return path
