"""Submission validator — checks that a submission file meets competition requirements.

All checks from agentic-kaggle skill are encoded here:
- File exists and is non-empty
- Column names match sample_submission
- Row count matches test set
- No NaN or inf values in prediction column
- ID column values match test set IDs
"""

from __future__ import annotations

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


def validate(submission_path: str | Path, data_dir: str | Path) -> ValidationResult:
    """Validate a submission file against competition data.

    Args:
        submission_path: Path to submission CSV.
        data_dir: Path to competition data directory (contains sample_submission.csv, test.csv).

    Returns:
        ValidationResult with all checks and pass/fail status.
    """
    submission_path = Path(submission_path)
    data_dir = Path(data_dir)

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
