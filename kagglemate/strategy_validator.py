"""Strategy validator — hardens LLM-generated feature strategies before they reach the template.

The validator treats the LLM output as a *suggestion*. It checks column existence,
prevents target/ID leakage, aligns numerical/categorical hints with actual dtypes,
verifies that feature-engineering code declares the columns it claims to create,
and optionally executes the FE code on the real train/test DataFrames. If the
strategy cannot be made valid, it falls back to a deterministic heuristic.
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path
from typing import Optional

import pandas as pd
from pydantic import BaseModel

from kagglemate.types import DataProfile


SUPPORTED_MODELS = {"lightgbm", "xgboost", "catboost", "lgbm", "xgb"}
DEFAULT_MODEL_PARAMS = {
    "lightgbm": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "num_leaves": 31,
        "random_state": 42,
        "verbose": -1,
    },
    "xgboost": {
        "n_estimators": 1000,
        "learning_rate": 0.05,
        "max_depth": 6,
        "random_state": 42,
    },
    "catboost": {
        "iterations": 1000,
        "learning_rate": 0.05,
        "depth": 6,
        "random_seed": 42,
        "silent": True,
    },
}


class StrategyValidationResult(BaseModel):
    valid: bool
    fixed: bool = False
    strategy: dict
    errors: list[str]
    warnings: list[str]
    issues: list[dict] = []


def validate_and_fix(
    strategy: dict,
    profile: DataProfile,
    train_df: pd.DataFrame | None = None,
    test_df: pd.DataFrame | None = None,
) -> StrategyValidationResult:
    """Validate and auto-fix an LLM feature strategy.

    Args:
        strategy: Dict with keys src_feature_cols, fe_feature_cols, numerical_cols,
            categorical_cols, feature_engineering, model_params, model_name.
        profile: DataProfile dict.
        train_df: Optional train DataFrame for FE code execution check.
        test_df: Optional test DataFrame for FE code execution check.

    Returns:
        StrategyValidationResult with the cleaned/fallback strategy and diagnostics.
    """
    strategy = dict(strategy)
    errors: list[str] = []
    warnings: list[str] = []
    issues: list[dict] = []

    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")
    all_columns = set(profile.get("columns", []))
    numerical_profile = set(profile.get("numerical_cols", []))
    categorical_profile = set(profile.get("categorical_cols", []))

    # ── 1. Normalize lists ──
    strategy["src_feature_cols"] = _as_list(strategy.get("src_feature_cols", []))
    strategy["fe_feature_cols"] = _as_list(strategy.get("fe_feature_cols", []))
    strategy["numerical_cols"] = _as_list(strategy.get("numerical_cols", []))
    strategy["categorical_cols"] = _as_list(strategy.get("categorical_cols", []))

    # ── 2. Feature cols must exist in train ──
    src_exists = [c for c in strategy["src_feature_cols"] if c in all_columns]
    missing_src = set(strategy["src_feature_cols"]) - set(src_exists)
    if missing_src:
        warnings.append(f"Removed non-existent source features: {sorted(missing_src)}")
        for col in sorted(missing_src):
            issues.append({"type": "nonexistent_column", "column": col, "action": "removed_from_features"})
        strategy["src_feature_cols"] = src_exists

    # ── 3. Prevent target / ID leakage ──
    forbidden = {id_col, target_col}
    for key in ("src_feature_cols", "fe_feature_cols", "numerical_cols", "categorical_cols"):
        before = set(strategy[key])
        strategy[key] = [c for c in strategy[key] if c not in forbidden]
        leaked = before - set(strategy[key])
        if leaked:
            warnings.append(f"Removed ID/target leakage from {key}: {sorted(leaked)}")
            issue_type = "id_leakage" if id_col in leaked else "target_leakage"
            for col in sorted(leaked):
                issues.append({"type": issue_type, "column": col, "action": "removed_from_features"})

    # ── 4. Align numerical / categorical with actual dtypes ──
    num_features = set(strategy["src_feature_cols"]) | set(strategy["fe_feature_cols"])
    fixed_numerical = []
    fixed_categorical = []
    for col in num_features:
        if col in numerical_profile:
            fixed_numerical.append(col)
        elif col in categorical_profile:
            fixed_categorical.append(col)
        else:
            # Column only exists in fe_feature_cols and dtype unknown until FE runs;
            # keep the original hint if present, otherwise default to numerical.
            if col in strategy["categorical_cols"]:
                fixed_categorical.append(col)
            else:
                fixed_numerical.append(col)

    moved_to_cat = set(strategy["numerical_cols"]) - set(fixed_numerical)
    moved_to_num = set(strategy["categorical_cols"]) - set(fixed_categorical)
    if moved_to_cat:
        warnings.append(f"Columns reclassified from numerical to categorical: {sorted(moved_to_cat)}")
    if moved_to_num:
        warnings.append(f"Columns reclassified from categorical to numerical: {sorted(moved_to_num)}")

    strategy["numerical_cols"] = fixed_numerical
    strategy["categorical_cols"] = fixed_categorical

    # ── 5. FE new columns should not already exist in source ──
    fe_cols = strategy["fe_feature_cols"]
    overlapping = set(fe_cols) & set(strategy["src_feature_cols"])
    if overlapping:
        warnings.append(
            f"fe_feature_cols overlap with src_feature_cols (will be overwritten): {sorted(overlapping)}"
        )

    # ── 6. Train/test schema compatibility ──
    if train_df is not None and test_df is not None:
        train_cols = set(train_df.columns)
        test_cols = set(test_df.columns)
        if train_cols - {target_col} != test_cols - {target_col}:
            warnings.append(
                f"Train/test schema mismatch: train={sorted(train_cols - {target_col})}, "
                f"test={sorted(test_cols - {target_col})}"
            )

    # ── 7. Duplicate feature columns ──
    all_feature_cols = strategy["src_feature_cols"] + strategy["fe_feature_cols"]
    seen = set()
    duplicates = []
    for col in all_feature_cols:
        if col in seen:
            duplicates.append(col)
        seen.add(col)
    if duplicates:
        warnings.append(f"Duplicate feature columns removed: {sorted(set(duplicates))}")
        strategy["src_feature_cols"] = [c for i, c in enumerate(strategy["src_feature_cols"]) if c not in duplicates or strategy["src_feature_cols"].index(c) == i]
        strategy["fe_feature_cols"] = [c for i, c in enumerate(strategy["fe_feature_cols"]) if c not in duplicates or strategy["fe_feature_cols"].index(c) == i]
        for col in sorted(set(duplicates)):
            issues.append({"type": "duplicate_column", "column": col, "action": "deduplicated"})

    # ── 8. High-cardinality categorical warning ──
    HIGH_CARDINALITY_THRESHOLD = 50
    for col in strategy["categorical_cols"]:
        col_detail = next((cd for cd in profile.get("column_details", []) if cd.get("name") == col), None)
        if col_detail and col_detail.get("n_unique", 0) > HIGH_CARDINALITY_THRESHOLD:
            warnings.append(
                f"High-cardinality categorical column '{col}' ({col_detail['n_unique']} unique values); "
                "consider target encoding or clustering instead of one-hot."
            )
            issues.append({"type": "high_cardinality", "column": col, "action": "warn_only"})

    # ── 9. Target-leakage column name heuristics ──
    # Exact target/id columns are already removed in step 3. Here we guard
    # against obvious posterior labels that should never be features, and warn
    # about columns whose names merely look suspicious.
    hard_posterior_keywords = {
        "ground_truth", "answer", "post_outcome", "label_encoded_target",
        "is_target", "target_label", "true_label",
    }
    suspicious_keywords = {"target", "label", "price", "survived", "transported"}

    for col in list(strategy["src_feature_cols"]):
        col_lower = col.lower()
        if any(kw in col_lower for kw in hard_posterior_keywords):
            warnings.append(f"Removed posterior label column '{col}' from features.")
            strategy["src_feature_cols"].remove(col)
            issues.append({"type": "target_leakage_heuristic", "column": col, "action": "removed_from_features"})
        elif col != target_col and any(kw in col_lower for kw in suspicious_keywords):
            warnings.append(
                f"Column '{col}' has a suspicious name but is not the target; "
                "review before using it as a feature."
            )
            issues.append({"type": "suspicious_column_name", "column": col, "action": "warn_only"})

    # ── 10. Constant or all-null columns ──
    if train_df is not None:
        for col in list(strategy["src_feature_cols"]):
            if col not in train_df.columns:
                continue
            n_unique = train_df[col].nunique(dropna=False)
            null_pct = train_df[col].isna().mean()
            if n_unique <= 1:
                warnings.append(f"Removed constant column '{col}'.")
                strategy["src_feature_cols"].remove(col)
                issues.append({"type": "constant_column", "column": col, "action": "removed_from_features"})
            elif null_pct >= 0.99:
                warnings.append(f"Removed near-all-null column '{col}' ({null_pct*100:.1f}% missing).")
                strategy["src_feature_cols"].remove(col)
                issues.append({"type": "all_null_column", "column": col, "action": "removed_from_features"})

    # ── 11. Feature engineering code checks ──
    fe_code = str(strategy.get("feature_engineering", "")).strip()
    if fe_code:
        ref_errors, ref_warnings = _check_fe_references(fe_code, all_columns, set(fe_cols))
        warnings.extend(ref_warnings)
        if ref_errors:
            # Auto-fix: drop the invalid FE block and continue with source features
            warnings.extend([f"Invalid feature engineering: {e}" for e in ref_errors])
            fe_code = ""
            strategy["fe_feature_cols"] = []

        # Optional: execute FE code on real data
        if fe_code and train_df is not None and test_df is not None:
            exec_ok, exec_msg = _try_execute_fe(fe_code, train_df, test_df, set(fe_cols))
            if not exec_ok:
                warnings.append(f"Feature engineering code execution failed: {exec_msg}")
                fe_code = ""
                strategy["fe_feature_cols"] = []
    else:
        strategy["fe_feature_cols"] = []

    strategy["feature_engineering"] = fe_code if fe_code else "# No custom feature engineering."

    # ── 12. Model name / params validation ──
    model_name = str(strategy.get("model_name", "LightGBM")).strip()
    normalized = model_name.lower().replace(" ", "").replace("_", "")
    if normalized not in SUPPORTED_MODELS:
        warnings.append(
            f"Unsupported model '{model_name}', falling back to LightGBM."
        )
        model_name = "LightGBM"
        normalized = "lightgbm"
    strategy["model_name"] = model_name

    params = strategy.get("model_params")
    if not isinstance(params, dict):
        warnings.append("model_params is not a dict; replaced with defaults.")
        params = DEFAULT_MODEL_PARAMS[normalized].copy()
    else:
        params = dict(params)
        # Ensure random_state / seed consistency
        if normalized in ("lightgbm", "lgbm") and "random_state" not in params:
            params["random_state"] = 42
        elif normalized in ("xgboost", "xgb") and "random_state" not in params:
            params["random_state"] = 42
        elif normalized == "catboost" and "random_seed" not in params:
            params["random_seed"] = 42
    strategy["model_params"] = params

    # ── 13. Recompute feature_cols and final empty check ──
    strategy["feature_cols"] = strategy["src_feature_cols"] + strategy["fe_feature_cols"]

    if not strategy["feature_cols"]:
        errors.append("No valid features after validation; falling back to heuristic strategy.")
        strategy = heuristic_strategy(profile)
        issues.append({"type": "fallback", "column": None, "action": "replaced_with_heuristic_strategy"})

    is_valid = len(errors) == 0 or "falling back" in errors[-1]
    fixed = len(issues) > 0 and is_valid
    return StrategyValidationResult(
        valid=is_valid,
        fixed=fixed,
        strategy=strategy,
        errors=errors,
        warnings=warnings,
        issues=issues,
    )


def heuristic_strategy(profile: DataProfile) -> dict:
    """Deterministic fallback strategy when LLM output is unusable."""
    all_cols = profile.get("columns", [])
    numerical = set(profile.get("numerical_cols", []))
    categorical = set(profile.get("categorical_cols", []))
    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")

    feature_cols = [c for c in all_cols if c not in (id_col, target_col) and c]
    num_features = [c for c in feature_cols if c in numerical]
    cat_features = [c for c in feature_cols if c in categorical]

    return {
        "feature_cols": feature_cols,
        "src_feature_cols": feature_cols,
        "fe_feature_cols": [],
        "numerical_cols": num_features,
        "categorical_cols": cat_features,
        "feature_engineering": "# No custom feature engineering for heuristic baseline.",
        "model_params": DEFAULT_MODEL_PARAMS["lightgbm"].copy(),
        "model_name": "LightGBM",
    }


def _as_list(value) -> list:
    if isinstance(value, list):
        return [str(c).strip() for c in value if str(c).strip()]
    if isinstance(value, str):
        return [value.strip()] if value.strip() else []
    return []


def _check_fe_references(code: str, all_columns: set[str], fe_cols: set[str]) -> tuple[list[str], list[str]]:
    """Check that FE code only references existing columns and declares fe_feature_cols."""
    errors: list[str] = []
    warnings: list[str] = []

    # Find train['col'] / test['col'] references
    all_refs = set(re.findall(r"(?:train|test)\[(?:'|\"|`)([^'\"`]+)(?:'|\"|`)]", code))
    # Columns being assigned are new columns and do not need to pre-exist
    assigned = set(re.findall(r"(?:train|test)\[(?:'|\"|`)([^'\"`]+)(?:'|\"|`)]\s*=", code))
    refs = all_refs - assigned
    bad_refs = refs - all_columns
    if bad_refs:
        errors.append(f"Feature engineering references non-existent columns: {sorted(bad_refs)}")

    missing_decl = fe_cols - assigned
    if missing_decl:
        warnings.append(
            f"fe_feature_cols not assigned in code: {sorted(missing_decl)}"
        )

    # Crude AST check for dangerous calls (optional hardening)
    try:
        tree = ast.parse(code)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import) or isinstance(node, ast.ImportFrom):
                module = getattr(node, "module", "")
                if module not in (None, "pandas", "numpy", "pd", "np"):
                    warnings.append(f"Feature engineering imports module '{module}'; ensure it is available.")
    except SyntaxError as e:
        errors.append(f"Feature engineering code has syntax error: {e}")

    return errors, warnings


def _try_execute_fe(
    code: str,
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    expected_cols: set[str],
) -> tuple[bool, str]:
    """Try to execute FE code in a restricted namespace and verify new columns exist."""
    try:
        namespace = {
            "pd": pd,
            "np": __import__("numpy"),
            "train": train_df.copy(),
            "test": test_df.copy(),
        }
        exec(code, namespace)
        train_out = namespace.get("train")
        test_out = namespace.get("test")
        if not isinstance(train_out, pd.DataFrame) or not isinstance(test_out, pd.DataFrame):
            return False, "Feature engineering did not leave 'train' and 'test' as DataFrames."

        missing_in_train = expected_cols - set(train_out.columns)
        missing_in_test = expected_cols - set(test_out.columns)
        if missing_in_train or missing_in_test:
            return False, f"Expected new columns missing: train={sorted(missing_in_train)}, test={sorted(missing_in_test)}"
        return True, "ok"
    except Exception as e:
        return False, str(e)


def save_validation_report(result: StrategyValidationResult, path: Path) -> Path:
    """Persist a strategy validation report as JSON.

    Args:
        result: StrategyValidationResult from validate_and_fix().
        path: Destination path (e.g. run_dir / "strategy_validation_report.json").

    Returns:
        The written path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "valid": result.valid,
        "fixed": result.fixed,
        "issues": result.issues,
        "warnings": result.warnings,
        "errors": result.errors,
        "final_feature_count": len(result.strategy.get("feature_cols", [])),
        "model_name": result.strategy.get("model_name"),
        "src_feature_count": len(result.strategy.get("src_feature_cols", [])),
        "fe_feature_count": len(result.strategy.get("fe_feature_cols", [])),
    }
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    return path
