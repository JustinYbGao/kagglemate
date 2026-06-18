"""Submission validator — checks that a submission file meets competition requirements.

All checks from agentic-kaggle skill are encoded here:
- File exists and is non-empty
- Column names match sample_submission
- Row count matches test set
- No NaN or inf values in prediction column
- ID column values match test set IDs
- Row order strictly matches sample_submission
- Probability values in [0, 1] for classification metrics
- Multiclass probability rows sum to ~1
- Duplicate submission hash detection
- Logloss metric clip reminder
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
from pydantic import BaseModel


class ValidationCheck(BaseModel):
    check: str
    passed: bool
    detail: str


class ValidationResult(BaseModel):
    is_valid: bool
    checks: list[ValidationCheck]
    errors: list[str]
    warnings: list[str]


def validate(
    submission_path: str | Path,
    data_dir: str | Path,
    metric: str | None = None,
    competition_slug: str | None = None,
) -> ValidationResult:
    """Validate a submission file against competition data.

    Args:
        submission_path: Path to submission CSV.
        data_dir: Path to competition data directory (contains sample_submission.csv, test.csv).
        metric: Optional evaluation metric name (e.g. "auc", "logloss", "rmse").
        competition_slug: Optional competition slug to look up previous submissions for
            duplicate detection.

    Returns:
        ValidationResult with all checks and pass/fail status.
    """
    submission_path = Path(submission_path)
    data_dir = Path(data_dir)
    metric = (metric or "").lower()

    checks: list[ValidationCheck] = []
    errors: list[str] = []
    warnings: list[str] = []

    # ── Check 1: File exists ──
    if not submission_path.exists():
        return ValidationResult(
            is_valid=False,
            checks=[ValidationCheck(check="file_exists", passed=False,
                                     detail=f"File not found: {submission_path}")],
            errors=[f"Submission file does not exist: {submission_path}"],
            warnings=[],
        )
    checks.append(ValidationCheck(check="file_exists", passed=True,
                                   detail=f"Found: {submission_path}"))

    # ── Check 2: File is non-empty and parseable ──
    try:
        sub = pd.read_csv(submission_path)
    except Exception as e:
        return ValidationResult(
            is_valid=False,
            checks=checks + [
                ValidationCheck(check="parseable_csv", passed=False,
                                 detail=f"Cannot parse CSV: {e}")
            ],
            errors=[f"Submission is not a valid CSV: {e}"],
            warnings=[],
        )

    if len(sub) == 0:
        checks.append(ValidationCheck(check="non_empty", passed=False,
                                       detail="Submission has 0 rows"))
        errors.append("Submission file is empty (0 rows)")
    else:
        checks.append(ValidationCheck(check="non_empty", passed=True,
                                       detail=f"{len(sub)} rows"))
    if len(sub.columns) == 0:
        checks.append(ValidationCheck(check="has_columns", passed=False,
                                       detail="Submission has 0 columns"))
        errors.append("Submission file has no columns")
    else:
        checks.append(ValidationCheck(check="has_columns", passed=True,
                                       detail=f"{len(sub.columns)} columns: {list(sub.columns)}"))

    # ── Check 3: Compare with sample submission ──
    sample = _find_sample(data_dir)
    col_ok, col_detail = _check_columns(sub, sample)
    checks.append(ValidationCheck(check="columns_match_sample", passed=col_ok,
                                   detail=col_detail))
    if not col_ok:
        errors.append(f"Column mismatch: {col_detail}")

    if sample is not None:
        row_ok, row_detail = _check_row_count(sub, sample)
        checks.append(ValidationCheck(check="row_count_matches", passed=row_ok,
                                       detail=row_detail))
        if not row_ok:
            errors.append(f"Row count mismatch: {row_detail}")

        id_ok, id_detail = _check_ids_match(sub, sample)
        checks.append(ValidationCheck(check="ids_match_test", passed=id_ok,
                                       detail=id_detail))
        if not id_ok:
            warnings.append(f"ID mismatch: {id_detail}")

        order_ok, order_detail = _check_row_order(sub, sample)
        checks.append(ValidationCheck(check="row_order_matches", passed=order_ok,
                                       detail=order_detail))
        if not order_ok:
            errors.append(f"Row order mismatch: {order_detail}")

    # ── Check 4: NaN and inf in prediction columns ──
    pred_col = _guess_prediction_column(sub)
    if pred_col:
        nan_count = int(sub[pred_col].isna().sum())
        inf_count = int(np.isinf(sub[pred_col].values).sum()) if sub[pred_col].dtype.kind in 'fc' else 0

        no_nan = nan_count == 0
        no_inf = inf_count == 0

        checks.append(ValidationCheck(check="no_nan", passed=no_nan,
                                       detail=f"{nan_count} NaN values in '{pred_col}'"))
        checks.append(ValidationCheck(check="no_inf", passed=no_inf,
                                       detail=f"{inf_count} inf values in '{pred_col}'"))

        if not no_nan:
            errors.append(f"Submission contains {nan_count} NaN values in prediction column")
        if not no_inf:
            errors.append(f"Submission contains {inf_count} infinite values in prediction column")
    else:
        warnings.append("Could not identify prediction column for NaN/inf check")

    # ── Check 5: Prediction values in reasonable range ──
    if pred_col:
        try:
            vals = sub[pred_col].dropna()
            if len(vals) > 0:
                vmin, vmax = float(vals.min()), float(vals.max())
                checks.append(ValidationCheck(
                    check="values_in_range", passed=True,
                    detail=f"Range: [{vmin:.4f}, {vmax:.4f}]"
                ))
                if vmin < -1e6 or vmax > 1e6:
                    warnings.append(f"Suspicious prediction range: [{vmin:.2f}, {vmax:.2f}]")
        except Exception:
            pass

    # ── Check 6: Probability-specific checks for classification metrics ──
    if pred_col and _looks_like_probability_metric(metric, sub, sample):
        vals = sub[pred_col].dropna()
        if len(vals) > 0:
            vmin, vmax = float(vals.min()), float(vals.max())
            prob_ok = vmin >= 0.0 and vmax <= 1.0
            checks.append(ValidationCheck(
                check="probability_range", passed=prob_ok,
                detail=f"Probability range: [{vmin:.4f}, {vmax:.4f}]"
            ))
            if not prob_ok:
                errors.append(
                    f"Probability predictions outside [0, 1]: [{vmin:.4f}, {vmax:.4f}]"
                )

    # ── Check 7: Multiclass probability sum ──
    pred_cols = _guess_prediction_columns(sub)
    if len(pred_cols) > 1:
        try:
            row_sums = sub[pred_cols].sum(axis=1)
            sums_ok = ((row_sums >= 0.99) & (row_sums <= 1.01)).all()
            checks.append(ValidationCheck(
                check="probability_row_sum", passed=bool(sums_ok),
                detail=f"Row sums ∈ [{float(row_sums.min()):.4f}, {float(row_sums.max()):.4f}]"
            ))
            if not sums_ok:
                errors.append(
                    f"Multiclass probability rows do not sum to 1.0: "
                    f"range [{float(row_sums.min()):.4f}, {float(row_sums.max()):.4f}]"
                )
        except Exception:
            pass

    # ── Check 8: Duplicate IDs ──
    if sample is not None:
        id_col = sample.columns[0] if len(sample.columns) > 0 else None
        if id_col and id_col in sub.columns:
            dup_count = int(sub[id_col].duplicated().sum())
            dup_ok = dup_count == 0
            checks.append(ValidationCheck(
                check="no_duplicate_ids", passed=dup_ok,
                detail=f"{dup_count} duplicate ID values"
            ))
            if not dup_ok:
                errors.append(f"Submission contains {dup_count} duplicate IDs")

    # ── Check 9: Binary classification prediction legality ──
    if pred_col and _is_binary_classification(metric, sample):
        vals = pd.to_numeric(sub[pred_col], errors="coerce").dropna()
        if len(vals) > 0:
            unique_vals = set(vals.unique())
            is_binary_labels = unique_vals.issubset({0, 1})
            is_probability = vals.min() >= 0 and vals.max() <= 1 and not is_binary_labels
            binary_ok = is_binary_labels or is_probability
            checks.append(ValidationCheck(
                check="binary_prediction_legal", passed=binary_ok,
                detail=f"unique values: {sorted(unique_vals)[:10]}"
            ))
            if not binary_ok:
                errors.append(
                    f"Binary classification predictions must be {{0,1}} labels or probabilities in [0,1]; "
                    f"got values like {sorted(unique_vals)[:5]}"
                )

    # ── Check 10: Regression extreme outliers ──
    if pred_col and _is_regression(metric):
        target_stats = _load_target_stats(data_dir)
        if target_stats is not None:
            vals = pd.to_numeric(sub[pred_col], errors="coerce").dropna()
            if len(vals) > 0:
                tmin, tmax = target_stats["min"], target_stats["max"]
                q1, q3 = target_stats["q1"], target_stats["q3"]
                iqr = q3 - q1
                lower = q1 - 5 * iqr if iqr > 0 else tmin
                upper = q3 + 5 * iqr if iqr > 0 else tmax
                out_of_range = ((vals < lower) | (vals > upper)).sum()
                outlier_ok = int(out_of_range) == 0
                checks.append(ValidationCheck(
                    check="regression_outliers", passed=outlier_ok,
                    detail=f"{int(out_of_range)} predictions outside training target range"
                ))
                if not outlier_ok:
                    warnings.append(
                        f"Regression predictions contain {int(out_of_range)} values far outside "
                        f"training target range [{tmin:.4f}, {tmax:.4f}]"
                    )

    # ── Check 11: Submission hash duplicate detection ──
    current_hash = hashlib.sha256(submission_path.read_bytes()).hexdigest()
    checks.append(ValidationCheck(
        check="submission_hash", passed=True,
        detail=f"sha256: {current_hash[:16]}..."
    ))
    if competition_slug:
        try:
            from kagglemate.memory.experiment_store import ExperimentStore
            store = ExperimentStore(competition_slug)
            previous_hashes = store.list_submission_hashes()
            if current_hash in previous_hashes:
                warnings.append(
                    "Duplicate submission detected: this file has the same hash as a previous submission."
                )
        except Exception as e:
            warnings.append(f"Could not check for duplicate submissions: {e}")

    # ── Check 13: Logloss clip reminder ──
    if pred_col and _is_logloss_like(metric):
        warnings.append(
            "Logloss-like metric detected: consider clipping probabilities with "
            "`np.clip(p, 1e-7, 1 - 1e-7)` to avoid log(0)."
        )

    is_valid = len(errors) == 0

    return ValidationResult(
        is_valid=is_valid,
        checks=checks,
        errors=errors,
        warnings=warnings,
    )


def _find_sample(data_dir: Path) -> Optional[pd.DataFrame]:
    """Find and read the sample submission file."""
    patterns = ["sample_submission", "gender_submission", "submission"]
    for pattern in patterns:
        for f in data_dir.glob("*.csv"):
            if pattern in f.name.lower():
                try:
                    return pd.read_csv(f)
                except Exception:
                    pass
    return None


def _check_columns(sub: pd.DataFrame, sample: Optional[pd.DataFrame]) -> tuple[bool, str]:
    """Check that submission columns match sample."""
    if sample is None:
        return True, "No sample submission to compare against"
    if list(sub.columns) == list(sample.columns):
        return True, "Columns match exactly"
    missing = set(sample.columns) - set(sub.columns)
    extra = set(sub.columns) - set(sample.columns)
    parts = []
    if missing:
        parts.append(f"missing: {missing}")
    if extra:
        parts.append(f"extra: {extra}")
    return False, "; ".join(parts)


def _check_row_count(sub: pd.DataFrame, sample: Optional[pd.DataFrame]) -> tuple[bool, str]:
    """Check row count against sample submission."""
    if sample is None:
        return True, "No sample to compare"
    expected = len(sample)
    actual = len(sub)
    if actual == expected:
        return True, f"{actual} rows (correct)"
    return False, f"Expected {expected}, got {actual}"


def _check_ids_match(sub: pd.DataFrame, sample: Optional[pd.DataFrame]) -> tuple[bool, str]:
    """Check that ID column values match (order-independent)."""
    if sample is None:
        return True, "No sample to compare"
    id_col = sample.columns[0] if len(sample.columns) > 0 else None
    if id_col is None or id_col not in sub.columns:
        return True, "Cannot determine ID column"
    sample_ids = set(sample[id_col].astype(str))
    sub_ids = set(sub[id_col].astype(str))
    if sample_ids == sub_ids:
        return True, "IDs match"
    missing = len(sample_ids - sub_ids)
    extra = len(sub_ids - sample_ids)
    return False, f"ID mismatch: {missing} missing, {extra} extra"


def _check_row_order(sub: pd.DataFrame, sample: Optional[pd.DataFrame]) -> tuple[bool, str]:
    """Check that ID column order strictly matches sample submission."""
    if sample is None:
        return True, "No sample to compare"
    id_col = sample.columns[0] if len(sample.columns) > 0 else None
    if id_col is None or id_col not in sub.columns:
        return True, "Cannot determine ID column"
    sample_order = list(sample[id_col].astype(str))
    sub_order = list(sub[id_col].astype(str))
    if sample_order == sub_order:
        return True, "Row order matches sample exactly"
    return False, "Row order differs from sample submission"


def _guess_prediction_column(sub: pd.DataFrame) -> Optional[str]:
    """Guess which column contains predictions (non-ID column)."""
    if len(sub.columns) == 1:
        return sub.columns[0]
    if len(sub.columns) == 2:
        for col in sub.columns:
            col_lower = col.lower()
            if "id" not in col_lower:
                return col
    # Last column heuristic
    return sub.columns[-1] if len(sub.columns) > 0 else None


def _guess_prediction_columns(sub: pd.DataFrame) -> list[str]:
    """Return all likely prediction columns (all columns except the ID column)."""
    if len(sub.columns) <= 1:
        return list(sub.columns)
    id_col = sub.columns[0]
    id_lower = id_col.lower()
    if "id" in id_lower:
        return [c for c in sub.columns if c != id_col]
    return [sub.columns[-1]]


def _looks_like_probability_metric(metric: str, sub: pd.DataFrame, sample: Optional[pd.DataFrame]) -> bool:
    """Return True if the metric or sample format suggests probability outputs."""
    regression_metrics = {"rmse", "mse", "mae", "rmsle", "r2", "mean_squared", "mean_absolute"}
    if any(m in metric for m in regression_metrics):
        return False
    probability_metrics = {"auc", "roc_auc", "logloss", "log_loss", "cross_entropy", "binary"}
    if any(m in metric for m in probability_metrics):
        return True
    # If sample values are in [0,1] and float-like, assume probabilities
    if sample is not None:
        pred_col = _guess_prediction_column(sample)
        if pred_col and pred_col in sample.columns:
            try:
                vals = pd.to_numeric(sample[pred_col], errors="coerce").dropna()
                if len(vals) > 0 and float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0:
                    return True
            except Exception:
                pass
    # If submission values are clearly probabilities
    pred_col = _guess_prediction_column(sub)
    if pred_col and pred_col in sub.columns:
        try:
            vals = pd.to_numeric(sub[pred_col], errors="coerce").dropna()
            if len(vals) > 0 and float(vals.min()) >= 0.0 and float(vals.max()) <= 1.0:
                return True
        except Exception:
            pass
    return False


def _is_logloss_like(metric: str) -> bool:
    """Return True for logloss / cross-entropy metrics."""
    return "log" in metric or "cross" in metric


def _is_binary_classification(metric: str, sample: Optional[pd.DataFrame]) -> bool:
    """Return True if the task appears to be binary classification."""
    regression_metrics = {"rmse", "mse", "mae", "rmsle", "r2", "mean_squared", "mean_absolute"}
    if any(m in metric for m in regression_metrics):
        return False
    binary_metrics = {"auc", "roc_auc", "logloss", "accuracy", "binary", "f1"}
    if any(m in metric for m in binary_metrics):
        return True
    if sample is not None and len(sample.columns) == 2:
        pred_col = _guess_prediction_column(sample)
        if pred_col and pred_col in sample.columns:
            vals = pd.to_numeric(sample[pred_col], errors="coerce").dropna()
            if len(vals) > 0 and set(vals.unique()).issubset({0, 1}):
                return True
    return False


def _is_regression(metric: str) -> bool:
    """Return True if the task appears to be regression."""
    regression_metrics = {"rmse", "mse", "mae", "rmsle", "r2", "mean_squared", "mean_absolute"}
    return any(m in metric for m in regression_metrics)


def _load_target_stats(data_dir: Path) -> Optional[dict]:
    """Load train.csv target column stats if available."""
    train_path = data_dir / "train.csv"
    if not train_path.exists():
        return None
    try:
        train = pd.read_csv(train_path)
        # Try common target column names
        target_candidates = [c for c in train.columns if any(kw in c.lower() for kw in ["target", "label", "survived", "saleprice", "transported"])]
        if not target_candidates:
            # Fallback: last column
            target_candidates = [train.columns[-1]]
        target_col = target_candidates[0]
        vals = pd.to_numeric(train[target_col], errors="coerce").dropna()
        if len(vals) == 0:
            return None
        return {
            "column": target_col,
            "min": float(vals.min()),
            "max": float(vals.max()),
            "q1": float(vals.quantile(0.25)),
            "q3": float(vals.quantile(0.75)),
        }
    except Exception:
        return None


def save_validation_report(result: ValidationResult, path: Path) -> Path:
    """Persist a submission validation report as JSON.

    Args:
        result: ValidationResult from validate().
        path: Destination path.

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "valid": result.is_valid,
        "errors": result.errors,
        "warnings": result.warnings,
        "checks": [c.model_dump() for c in result.checks],
        "submission_hash": next(
            (c.detail.replace("sha256: ", "").replace("...", "") for c in result.checks if c.check == "submission_hash"),
            "",
        ),
        "row_count": next(
            (int(c.detail.split()[0]) for c in result.checks if c.check == "non_empty" and c.passed),
            None,
        ),
        "columns": [],
    }
    cols_check = next((c for c in result.checks if c.check == "has_columns" and c.passed), None)
    if cols_check:
        try:
            report["columns"] = cols_check.detail.split(": ")[1].strip("[]").replace("'", "").split(", ")
        except Exception:
            pass
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
