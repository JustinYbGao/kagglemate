"""Baseline Node — generates a training script using Jinja2 template + LLM feature engineering.

Strategy:
1. LLM picks feature columns, feature engineering code, and model params
2. Jinja2 renders the full script from a battle-tested template
3. Template handles: data loading, CV loop, imputation, submission generation
4. LLM only writes the creative part (feature engineering)

This split gives us: reliability (template) + flexibility (LLM).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.llm_client import simple_prompt


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
"""


def run(state: KaggleAgentState) -> dict:
    """Generate a baseline training script.

    Returns state updates with the generated script path and experiment metadata.
    """
    profile = state.get("data_profile") or {}
    comp_type = state.get("competition_type", "tabular_classification")

    _log(f"Designing baseline for: {state['competition_slug']} ({comp_type})")

    # ── Step 1: LLM decides feature strategy ──
    strategy = _get_strategy(state, profile)

    # ── Step 2: Render the training script ──
    script_path = _render_script(state, profile, strategy)

    # ── Step 3: Build experiment record ──
    experiment = {
        "name": f"baseline_{strategy.get('model_name', 'lgbm')}_001",
        "model": strategy.get("model_name", "LightGBM"),
        "cv_score": 0.0,
        "cv_std": 0.0,
        "lb_score": None,
        "metric": state.get("evaluation_metric", "unknown"),
        "params": strategy.get("model_params", {}),
        "features": strategy.get("feature_cols", []),
        "submission_path": "",
        "script_path": str(script_path),
        "status": "pending",
    }

    return {
        "current_experiment": experiment,
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
        strategy = _heuristic_strategy(profile)

    # Normalize: support both old "feature_cols" and new "src_feature_cols"/"fe_feature_cols"
    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")
    if "src_feature_cols" not in strategy:
        all_features = strategy.get("feature_cols", [])
        strategy["src_feature_cols"] = [c for c in all_features if c != id_col and c != target_col]
        strategy["fe_feature_cols"] = []
    if "feature_cols" not in strategy:
        strategy["feature_cols"] = strategy["src_feature_cols"] + strategy.get("fe_feature_cols", [])

    # Filter out ID/target from both lists
    strategy["src_feature_cols"] = [c for c in strategy.get("src_feature_cols", []) if c not in (id_col, target_col)]
    strategy["fe_feature_cols"] = [c for c in strategy.get("fe_feature_cols", []) if c not in (id_col, target_col)]
    strategy["feature_cols"] = strategy["src_feature_cols"] + strategy["fe_feature_cols"]

    # Default model name
    if not strategy.get("model_name"):
        comp_type = state.get("competition_type", "")
        strategy["model_name"] = (
            "LightGBM" if "classif" in comp_type else "XGBRegressor"
        )

    return strategy


def _heuristic_strategy(profile: dict) -> dict:
    """Fallback: pick features without LLM."""
    all_cols = profile.get("columns", [])
    numerical = profile.get("numerical_cols", [])
    categorical = profile.get("categorical_cols", [])
    id_col = profile.get("id_col", "")
    target_col = profile.get("target_col", "")

    feature_cols = [
        c for c in all_cols if c != id_col and c != target_col
    ]
    num_features = [c for c in feature_cols if c in numerical]
    cat_features = [c for c in feature_cols if c in categorical]

    return {
        "feature_cols": feature_cols,
        "numerical_cols": num_features,
        "categorical_cols": cat_features,
        "feature_engineering": (
            "# No feature engineering for heuristic baseline.\n"
            "    # Add custom feature engineering here."
        ),
        "model_params": {
            "n_estimators": 1000,
            "learning_rate": 0.05,
            "num_leaves": 31,
            "random_state": 42,
            "verbose": -1,
        },
        "model_name": "LightGBM",
    }


def _format_column_details(profile: dict) -> str:
    """Format column info as a table for the LLM prompt."""
    lines = ["| Column | Dtype | Missing% | Unique |"]
    lines.append("|--------|-------|----------|--------|")
    for cd in profile.get("column_details", []):
        lines.append(
            f"| {cd['name']} | {cd['dtype']} | {cd['missing_pct']}% | {cd['n_unique']} |"
        )
    return "\n".join(lines)


def _render_script(state: KaggleAgentState, profile: dict, strategy: dict) -> Path:
    """Render the Jinja2 template and save the training script."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent.parent.parent / "templates"))
    )
    template = env.get_template("baseline_script_template.py.j2")

    comp_type = state.get("competition_type", "tabular_classification")
    is_classification = "classif" in comp_type.lower() or "binary" in comp_type.lower()
    is_multiclass = "multi" in comp_type.lower()

    metric_name = state.get("evaluation_metric", "auc").lower()

    # Map metric to sklearn function + import
    metric_map = _get_metric_config(metric_name, is_classification)

    # CV setup
    cv_setup, cv_import = _get_cv_config(is_classification, is_multiclass)

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
        cv_strategy=f"{cv_import.rsplit('.', 1)[-1]} N_FOLDS-fold",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_dir=state.get("data_dir", ""),
        target_col=profile.get("target_col", "target"),
        id_col=profile.get("id_col", "id"),
        submission_cols=profile.get("submission_cols", []),
        submission_target_col=sub_target,
        n_folds=5,
        cv_import=cv_import,
        cv_setup=cv_setup,
        metric_import=metric_map["import"],
        metric_call=metric_map["call"],
        metric_name=metric_name.upper(),
        is_classification=is_classification,
        src_feature_cols=json.dumps(strategy.get("src_feature_cols", [])),
        fe_feature_cols=json.dumps(strategy.get("fe_feature_cols", [])),
        numerical_cols=json.dumps(strategy.get("numerical_cols", [])),
        categorical_cols=json.dumps(strategy.get("categorical_cols", [])),
        feature_engineering=_normalize_indent(strategy.get("feature_engineering", "")),
        model_params=json.dumps(strategy.get("model_params", {}), indent=4),
        model_init=model_init,
        model_import=model_import,
        submission_dir=str(Path(state.get("submission_dir", ""))),
        submission_filename=sub_name,
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


def _get_cv_config(is_classification: bool, is_multiclass: bool) -> tuple[str, str]:
    if is_classification:
        # For multiclass with many classes use KFold, otherwise StratifiedKFold
        if is_multiclass:
            return "KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)", "KFold"
        return "StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)", "StratifiedKFold"
    return "KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_SEED)", "KFold"


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
