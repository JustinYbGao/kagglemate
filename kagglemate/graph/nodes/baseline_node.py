"""Baseline Node — generates a training script using Jinja2 template + validated LLM suggestions.

Strategy:
1. CV strategy is decided deterministically by cv_strategy.py (no LLM).
2. LLM suggests feature columns, feature engineering code, and model params.
3. strategy_validator.py hardens the suggestion: removes bad columns, fixes dtype
   mismatches, optionally executes FE code, and falls back to a heuristic if needed.
4. The validated strategy + CV plan are written to experiment_config.json.
5. Jinja2 renders the full script from a battle-tested template.

This split gives us: reliability (template + deterministic CV + validation) +
flexibility (LLM feature engineering suggestions).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from datetime import datetime, timezone

import pandas as pd

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.llm_client import simple_prompt
from kagglemate.cv_strategy import generate_cv_plan
from kagglemate.strategy_validator import validate_and_fix, heuristic_strategy, save_validation_report


# ── LLM prompt: decide features and strategy ──

FEATURE_STRATEGY_PROMPT = """You are a Kaggle ML engineer. Based on the data profile below, design the feature strategy for a baseline model.

## Competition
- Type: {competition_type}
- Metric: {evaluation_metric}
- Target column: {target_col}
- ID column: {id_col}
- Train shape: {train_rows} rows × {train_cols} cols
- Test shape: {test_rows} rows
- Submission columns: {submission_cols}

## Column Details
{column_details}

## Your Job
1. Pick which columns to use as FEATURES (exclude ID column and target column)
2. Identify which features are NUMERICAL vs CATEGORICAL
3. Write a SHORT feature engineering block (3-8 lines of pandas code). Keep it simple for baseline:
   - Maybe create 1-2 interaction features
   - Maybe extract title from Name column (Titanic)
   - Maybe bin ages or fares
   - Maybe create family size (SibSp + Parch)
   - Don't over-engineer — this is a BASELINE
4. Pick reasonable model hyperparameters

Output as JSON:
```json
{{
  "src_feature_cols": ["col1", "col2", ...],
  "fe_feature_cols": ["new_col_from_fe", ...],
  "numerical_cols": ["col1", "col3", ...],
  "categorical_cols": ["col2", "col4", ...],
  "feature_engineering": "# pandas code here\\n# Use train['col'] and test['col']",
  "model_params": {{"n_estimators": 1000, "learning_rate": 0.05, ...}},
  "model_name": "LightGBM"
}}
```

IMPORTANT:
- src_feature_cols: columns from the original CSV that should be used as features (exclude ID and target)
- fe_feature_cols: NEW columns created by your feature_engineering code (e.g., "Title", "FamilySize")
- If a column has >50 unique values and is object/string type, consider it categorical but note high cardinality
- For classification, predict class probabilities. For regression, predict values.
- The feature_engineering code works on the FULL pandas DataFrames `train` and `test` (not X/X_test)
- Use train['col_name'] and test['col_name'] syntax
- You can create NEW columns: train['NewCol'] = train['OldCol'] * 2
- List any NEW columns you create in fe_feature_cols (source columns stay in src_feature_cols)
- Do NOT reference columns that do not exist in the data profile
"""


def run(state: KaggleAgentState) -> dict:
    """Generate a baseline training script.

    Returns state updates with the generated script path and experiment metadata.
    """
    profile = state.get("data_profile") or {}
    comp_type = state.get("competition_type", "tabular_classification")
    metric = state.get("evaluation_metric", "unknown")
    target_col = profile.get("target_col", "target")

    _log(f"Designing baseline for: {state['competition_slug']} ({comp_type})")

    # ── Step 1: Deterministic CV plan (read before generating script) ──
    cv_plan = generate_cv_plan(
        profile,
        {"slug": state["competition_slug"], "type": comp_type},
        metric,
        target_col,
        state.get("report_dir", ""),
    )
    _log(f"CV plan: {cv_plan['strategy']} — {cv_plan['reasoning']}")
    if cv_plan.get("risk_notes"):
        for note in cv_plan["risk_notes"]:
            _log(f"CV risk: {note}")

    # ── Step 2: LLM suggests feature strategy (not trusted yet) ──
    strategy = _get_strategy(state, profile)

    # ── Step 3: Validate and harden the LLM suggestion ──
    train_df, test_df = _load_train_test(state, profile)
    val_result = validate_and_fix(strategy, profile, train_df, test_df)
    strategy = val_result.strategy

    if val_result.warnings:
        for w in val_result.warnings:
            _log(f"Strategy warning: {w}")
    if val_result.errors:
        for e in val_result.errors:
            _log(f"Strategy error: {e}")

    # ── Step 4: Persist experiment config and strategy validation report ──
    config_path = _write_experiment_config(state, profile, strategy, cv_plan, val_result)
    report_path = _write_strategy_validation_report(val_result, config_path)

    # ── Step 5: Render the training script ──
    script_path = _render_script(state, profile, strategy, cv_plan, config_path)

    # ── Step 6: Build experiment record ──
    experiment = {
        "name": f"baseline_{strategy.get('model_name', 'lgbm').lower().replace(' ', '_')}_001",
        "model": strategy.get("model_name", "LightGBM"),
        "task_type": comp_type,
        "target_column": profile.get("target_col", ""),
        "id_column": profile.get("id_col", ""),
        "cv_strategy": cv_plan.get("strategy", ""),
        "cv_score": 0.0,
        "cv_std": 0.0,
        "lb_score": None,
        "metric": metric,
        "params": strategy.get("model_params", {}),
        "features": strategy.get("feature_cols", []),
        "submission_path": "",
        "script_path": str(script_path),
        "config_path": str(config_path),
        "strategy_validation_report_path": str(report_path),
        "status": "pending",
    }

    return {
        "current_experiment": experiment,
        "cv_plan": cv_plan,
        "current_phase": "build",
    }


def _get_strategy(state: KaggleAgentState, profile: dict) -> dict:
    """Get feature strategy from LLM."""
    column_details = _format_column_details(profile)

    prompt = FEATURE_STRATEGY_PROMPT.format(
        competition_type=state.get("competition_type", "unknown"),
        evaluation_metric=state.get("evaluation_metric", "unknown"),
        target_col=profile.get("target_col", "?"),
        id_col=profile.get("id_col", "?"),
        train_rows=profile.get("train_rows", "?"),
        train_cols=len(profile.get("columns", [])),
        test_rows=profile.get("test_rows", "?"),
        submission_cols=json.dumps(profile.get("submission_cols", [])),
        column_details=column_details,
    )

    try:
        raw = simple_prompt(prompt)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        strategy = json.loads(raw.strip())
    except Exception as e:
        _log(f"LLM strategy failed: {e}. Using heuristic fallback.")
        strategy = heuristic_strategy(profile)

    # Normalize: support both old "feature_cols" and new "src_feature_cols"/"fe_feature_cols"
    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")
    if "src_feature_cols" not in strategy:
        all_features = strategy.get("feature_cols", [])
        strategy["src_feature_cols"] = [c for c in all_features if c != id_col and c != target_col]
        strategy["fe_feature_cols"] = []
    if "feature_cols" not in strategy:
        strategy["feature_cols"] = strategy["src_feature_cols"] + strategy.get("fe_feature_cols", [])

    # Default model name
    if not strategy.get("model_name"):
        comp_type = state.get("competition_type", "")
        strategy["model_name"] = (
            "LightGBM" if "classif" in comp_type else "XGBRegressor"
        )

    return strategy


def _load_train_test(state: KaggleAgentState, profile: dict) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """Load train/test CSVs for FE code validation."""
    data_dir = Path(state.get("data_dir", ""))
    if not data_dir.exists():
        return None, None

    train_file = _find_csv(profile, str(data_dir), "train")
    test_file = _find_csv(profile, str(data_dir), "test")

    train_path = data_dir / train_file if train_file else None
    test_path = data_dir / test_file if test_file else None

    train_df: pd.DataFrame | None = None
    test_df: pd.DataFrame | None = None
    try:
        if train_path and train_path.exists():
            train_df = pd.read_csv(train_path)
            train_df.columns = train_df.columns.str.strip()
    except Exception as e:
        _log(f"Could not load train CSV for validation: {e}")
    try:
        if test_path and test_path.exists():
            test_df = pd.read_csv(test_path)
            test_df.columns = test_df.columns.str.strip()
    except Exception as e:
        _log(f"Could not load test CSV for validation: {e}")

    return train_df, test_df


def _write_strategy_validation_report(val_result, config_path: Path) -> Path:
    """Write strategy validation report next to experiment config."""
    from kagglemate.strategy_validator import save_validation_report
    report_path = config_path.parent / "strategy_validation_report.json"
    save_validation_report(val_result, report_path)
    _log(f"Saved strategy validation report → {report_path}")
    return report_path


def _write_experiment_config(
    state: KaggleAgentState,
    profile: dict,
    strategy: dict,
    cv_plan: dict,
    val_result,
) -> Path:
    """Write a JSON config that fully describes the experiment for reproducibility."""
    script_dir = state.get("script_dir", "")
    if not script_dir:
        script_dir = Path(state.get("data_dir", "")).parent.parent / "scripts"
    script_dir = Path(script_dir)
    script_dir.mkdir(parents=True, exist_ok=True)

    prompt_summary = FEATURE_STRATEGY_PROMPT[:500].replace("\n", " ")

    config_data = {
        "competition_slug": state.get("competition_slug", ""),
        "competition_type": state.get("competition_type", ""),
        "metric": state.get("evaluation_metric", "unknown"),
        "target_col": profile.get("target_col", ""),
        "id_col": profile.get("id_col", ""),
        "model_name": strategy.get("model_name", "LightGBM"),
        "model_params": strategy.get("model_params", {}),
        "cv_plan": cv_plan,
        "src_feature_cols": strategy.get("src_feature_cols", []),
        "fe_feature_cols": strategy.get("fe_feature_cols", []),
        "numerical_cols": strategy.get("numerical_cols", []),
        "categorical_cols": strategy.get("categorical_cols", []),
        "feature_cols": strategy.get("feature_cols", []),
        "feature_engineering": strategy.get("feature_engineering", ""),
        "seed": cv_plan.get("random_seed", 42),
        "prompt_summary": prompt_summary,
        "validation_warnings": val_result.warnings,
        "validation_errors": val_result.errors,
        "generated_at": datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
    }

    out = script_dir / "experiment_config.json"
    out.write_text(json.dumps(config_data, indent=2, ensure_ascii=False), encoding="utf-8")
    _log(f"Saved experiment config → {out}")
    return out


def _render_script(state: KaggleAgentState, profile: dict, strategy: dict, cv_plan: dict, config_path: Path) -> Path:
    """Render the Jinja2 template and save the training script."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent.parent.parent / "templates"))
    )
    template = env.get_template("baseline_script_template.py.j2")

    comp_type = state.get("competition_type", "tabular_classification")
    is_classification = cv_plan.get("is_classification", True)
    is_multiclass = cv_plan.get("n_classes", 2) > 2

    metric_name = state.get("evaluation_metric", "auc").lower()
    # For binary/multiclass classification, output probabilities when the metric
    # expects probabilities (auc / logloss / cross_entropy); otherwise output
    # class labels (argmax) for accuracy-style metrics.
    output_probability = is_classification and (
        "auc" in metric_name or "log" in metric_name or "cross" in metric_name
    )

    # Map metric to sklearn function + import
    metric_map = _get_metric_config(metric_name, is_classification)

    # Model init
    model_name = strategy.get("model_name", "LightGBM")
    model_init, model_import = _get_model_config(model_name, is_classification)

    # Submission filename
    sub_name = f"baseline_{model_name.lower().replace(' ', '_')}_001.csv"

    # Determine submission target column (the prediction column name)
    sub_cols = profile.get("submission_cols", [])
    sub_target = sub_cols[-1] if len(sub_cols) >= 2 else (profile.get("target_col", "prediction"))

    script_dir = state.get("script_dir", "")
    if not script_dir:
        script_dir = Path(state.get("data_dir", "")).parent.parent / "scripts"

    content = template.render(
        competition_slug=state["competition_slug"],
        competition_type=comp_type,
        model_name=model_name,
        cv_strategy=cv_plan["strategy"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_dir=str(Path(state.get("data_dir", "")).resolve()),
        target_col=profile.get("target_col", "target"),
        id_col=profile.get("id_col", "id"),
        submission_cols=profile.get("submission_cols", []),
        submission_target_col=sub_target,
        n_folds=cv_plan.get("n_folds", 5),
        random_seed=cv_plan.get("random_seed", 42),
        cv_import=cv_plan["cv_import"],
        cv_setup=cv_plan["cv_setup"],
        cv_split_args=cv_plan["cv_split_args"],
        group_col=cv_plan.get("group_col"),
        date_col=cv_plan.get("date_col"),
        metric_import=metric_map["import"],
        metric_call=metric_map["call"],
        metric_name=metric_name.upper(),
        is_classification=is_classification,
        is_multiclass=is_multiclass,
        output_probability=output_probability,
        multiclass_submission_cols=json.dumps(sub_cols[1:] if len(sub_cols) > 2 else []),
        src_feature_cols=json.dumps(strategy.get("src_feature_cols", [])),
        fe_feature_cols=json.dumps(strategy.get("fe_feature_cols", [])),
        numerical_cols=json.dumps(strategy.get("numerical_cols", [])),
        categorical_cols=json.dumps(strategy.get("categorical_cols", [])),
        feature_engineering=_normalize_indent(strategy.get("feature_engineering", "")),
        model_params=json.dumps(strategy.get("model_params", {}), indent=4),
        model_init=model_init,
        model_import=model_import,
        submission_dir=str(Path(state.get("submission_dir", "")).resolve()),
        submission_filename=sub_name,
        config_path=str(config_path),
        train_file=_find_csv(profile, state.get("data_dir", ""), "train"),
        test_file=_find_csv(profile, state.get("data_dir", ""), "test"),
    )

    out = Path(script_dir) / f"train_baseline_{model_name.lower().replace(' ', '_')}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    _log(f"Saved baseline script → {out}")
    return out


def _find_csv(profile: dict, data_dir: str, pattern: str) -> str:
    """Find the actual filename for train/test CSV in data_dir."""
    d = Path(data_dir)
    if d.exists():
        for f in d.glob("*.csv"):
            if pattern.lower() in f.name.lower():
                return f.name
    return f"{pattern}.csv"


def _get_metric_config(metric: str, is_classification: bool) -> dict:
    """Map metric name to sklearn import and call."""
    if is_classification:
        if "auc" in metric or "roc" in metric:
            return {
                "import": "roc_auc_score",
                "call": "roc_auc_score(y_val, model.predict_proba(X_val)[:, 1]) "
                        "if model.predict_proba(X_val).shape[1] == 2 "
                        "else roc_auc_score(y_val, model.predict_proba(X_val), multi_class='ovr')",
            }
        elif "log" in metric or "cross" in metric:
            return {
                "import": "log_loss",
                "call": "log_loss(y_val, model.predict_proba(X_val))",
            }
        elif "f1" in metric:
            return {
                "import": "f1_score",
                "call": "f1_score(y_val, model.predict(X_val), average='weighted')",
            }
        else:  # accuracy
            return {
                "import": "accuracy_score",
                "call": "accuracy_score(y_val, model.predict(X_val))",
            }
    else:
        if "rmse" in metric:
            return {
                "import": "mean_squared_error",
                "call": "np.sqrt(mean_squared_error(y_val, model.predict(X_val)))",
            }
        elif "mae" in metric:
            return {
                "import": "mean_absolute_error",
                "call": "mean_absolute_error(y_val, model.predict(X_val))",
            }
        else:
            return {
                "import": "mean_squared_error",
                "call": "np.sqrt(mean_squared_error(y_val, model.predict(X_val)))",
            }


def _get_model_config(model_name: str, is_classification: bool) -> tuple[str, str]:
    """Map model name to init code and import."""
    model_lower = model_name.lower().replace(" ", "").replace("_", "")

    if "xgb" in model_lower or "xgboost" in model_lower:
        if is_classification:
            return "XGBClassifier(**params)", "from xgboost import XGBClassifier"
        return "XGBRegressor(**params)", "from xgboost import XGBRegressor"
    elif "cat" in model_lower or "catboost" in model_lower:
        if is_classification:
            return "CatBoostClassifier(**params, silent=True)", "from catboost import CatBoostClassifier"
        return "CatBoostRegressor(**params, silent=True)", "from catboost import CatBoostRegressor"
    else:  # LightGBM default
        if is_classification:
            return "LGBMClassifier(**params)", "from lightgbm import LGBMClassifier"
        return "LGBMRegressor(**params)", "from lightgbm import LGBMRegressor"


def _format_column_details(profile: dict) -> str:
    """Format column info as a table for the LLM prompt."""
    lines = ["| Column | Dtype | Missing% | Unique |"]
    lines.append("|--------|-------|----------|--------|")
    for cd in profile.get("column_details", []):
        lines.append(
            f"| {cd['name']} | {cd['dtype']} | {cd['missing_pct']}% | {cd['n_unique']} |"
        )
    return "\n".join(lines)


def _normalize_indent(code: str) -> str:
    """Normalize LLM-generated code indentation for template insertion.

    The template expects 4-space-indented code at the point of insertion.
    Aggressively strips all leading whitespace per line, then re-indents.
    """
    code = code.strip()
    if not code:
        return "    # No custom feature engineering.\n    pass"
    # Strip all leading whitespace from each line, then indent 4 spaces
    lines = [line.strip() for line in code.split("\n")]
    return "\n".join(f"    {line}" if line else "" for line in lines)


def _log(msg: str):
    print(f"  [baseline] {msg}")
