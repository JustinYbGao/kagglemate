"""Tune Node — Optuna hyperparameter optimization.

Generates a tuning script that uses Optuna to search hyperparameters,
retrains with the best params, and outputs a submission file.

Works for: LightGBM, XGBoost, CatBoost (tabular classification & regression).
"""

from __future__ import annotations

import json
import textwrap
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.llm_client import simple_prompt


# ── Optuna search space definitions per model ──

LGBM_SEARCH_SPACE = {
    "n_estimators":        {"type": "int", "low": 100, "high": 3000, "log": False},
    "learning_rate":       {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "num_leaves":          {"type": "int", "low": 8, "high": 256, "log": False},
    "max_depth":           {"type": "int", "low": 3, "high": 16, "log": False},
    "min_child_samples":   {"type": "int", "low": 5, "high": 200, "log": True},
    "subsample":           {"type": "float", "low": 0.5, "high": 1.0, "log": False},
    "colsample_bytree":    {"type": "float", "low": 0.3, "high": 1.0, "log": False},
    "reg_alpha":           {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda":          {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}

XGB_SEARCH_SPACE = {
    "n_estimators":        {"type": "int", "low": 100, "high": 3000, "log": False},
    "learning_rate":       {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "max_depth":           {"type": "int", "low": 3, "high": 16, "log": False},
    "min_child_weight":    {"type": "float", "low": 0.1, "high": 50.0, "log": True},
    "subsample":           {"type": "float", "low": 0.5, "high": 1.0, "log": False},
    "colsample_bytree":    {"type": "float", "low": 0.3, "high": 1.0, "log": False},
    "gamma":               {"type": "float", "low": 1e-8, "high": 5.0, "log": True},
    "reg_alpha":           {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
    "reg_lambda":          {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}

CATBOOST_SEARCH_SPACE = {
    "iterations":          {"type": "int", "low": 100, "high": 3000, "log": False},
    "learning_rate":       {"type": "float", "low": 0.005, "high": 0.3, "log": True},
    "depth":               {"type": "int", "low": 3, "high": 12, "log": False},
    "l2_leaf_reg":         {"type": "float", "low": 0.1, "high": 50.0, "log": True},
    "border_count":        {"type": "int", "low": 32, "high": 255, "log": False},
    "random_strength":     {"type": "float", "low": 1e-8, "high": 10.0, "log": True},
}


SEARCH_SPACES = {
    "lgbm": LGBM_SEARCH_SPACE,
    "lightgbm": LGBM_SEARCH_SPACE,
    "xgb": XGB_SEARCH_SPACE,
    "xgboost": XGB_SEARCH_SPACE,
    "cat": CATBOOST_SEARCH_SPACE,
    "catboost": CATBOOST_SEARCH_SPACE,
}


def run(state: KaggleAgentState) -> dict:
    """Generate a hyperparameter tuning script.

    Uses the same strategy as baseline_node but adds Optuna optimization.
    """
    profile = state.get("data_profile") or {}
    comp_type = state.get("competition_type", "tabular_classification")
    slug = state["competition_slug"]

    n_trials = state.get("tune_trials", 50)

    _log(f"Designing tuning script for {slug} ({comp_type}, {n_trials} trials)")

    # ── Step 1: Get feature strategy (reuse baseline_node's logic) ──
    from kagglemate.graph.nodes.baseline_node import _get_strategy, _normalize_indent, \
        _get_metric_config, _get_cv_config, _get_model_config

    strategy = _get_strategy(state, profile)
    model_name = strategy.get("model_name", "LightGBM")
    model_lower = model_name.lower().replace(" ", "").replace("_", "")

    # Determine search space
    search_space = SEARCH_SPACES.get(model_lower, LGBM_SEARCH_SPACE)

    # ── Step 2: Render tuning script ──
    script_path = _render_tune_script(
        state, profile, strategy, search_space, n_trials, model_name, model_lower,
        _get_metric_config, _get_cv_config, _get_model_config,
    )

    # ── Step 3: Build experiment record ──
    experiment = {
        "name": f"tune_{model_lower}_{n_trials}trials",
        "model": model_name,
        "cv_score": 0.0,
        "lb_score": None,
        "metric": state.get("evaluation_metric", "unknown"),
        "features": strategy.get("feature_cols", []),
        "submission_path": "",
        "script_path": str(script_path),
        "status": "pending",
    }

    return {
        "current_experiment": experiment,
        "current_phase": "build",
    }


def _render_tune_script(state, profile, strategy, search_space, n_trials, model_name, model_lower, _get_metric_config, _get_cv_config, _get_model_config) -> Path:
    """Render the Optuna tuning Jinja2 template."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(loader=FileSystemLoader(
        str(Path(__file__).parent.parent.parent / "templates")))
    template = env.get_template("tune_script_template.py.j2")

    comp_type = state.get("competition_type", "tabular_classification")
    is_classification = "classif" in comp_type.lower()
    metric_name = state.get("evaluation_metric", "auc").lower()
    metric_config = _get_metric_config(metric_name, is_classification)
    cv_setup, cv_import = _get_cv_config(is_classification, "multi" in comp_type.lower())
    model_init, model_import = _get_model_config(model_name, is_classification)

    # Tune-specific: model init for Optuna objective vs retrain loop
    # objective uses `params` dict from trial; retrain uses `**best_params`
    model_class = model_init.split("(")[0].strip()  # "LGBMClassifier"
    model_init_tune = model_init
    model_init_best = model_init  # Not used directly in retrain — uses model_class

    # Direction: maximize for most metrics, minimize for RMSE/MAE/logloss
    optuna_direction = "minimize" if any(m in metric_name for m in ["rmse", "mae", "log", "mse"]) else "maximize"

    sub_name = f"tuned_{model_name.lower().replace(' ', '_')}_{n_trials}trials.csv"
    sub_cols = profile.get("submission_cols", [])
    sub_target = sub_cols[-1] if len(sub_cols) >= 2 else (profile.get("target_col", "prediction"))

    script_dir = state.get("script_dir", "")
    if not script_dir:
        script_dir = Path(state.get("data_dir", "")).parent.parent / "scripts"

    # Metric call for Optuna objective (uses trial params)
    tune_call = metric_config["call"]
    # For final eval, use same call
    final_call = tune_call

    content = template.render(
        competition_slug=state["competition_slug"],
        competition_type=comp_type,
        model_name=model_name,
        cv_strategy=f"{cv_import.split('.')[-1]} N_FOLDS-fold",
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        data_dir=state.get("data_dir", ""),
        target_col=profile.get("target_col", "target"),
        id_col=profile.get("id_col", "id"),
        submission_cols=profile.get("submission_cols", []),
        submission_target_col=sub_target,
        n_folds=5,
        n_trials=n_trials,
        cv_import=cv_import,
        cv_setup=cv_setup,
        metric_import=metric_config["import"],
        metric_tune_call=tune_call,
        metric_final_call=final_call,
        metric_name=metric_name.upper(),
        optuna_direction=optuna_direction,
        is_classification=is_classification,
        src_feature_cols=json.dumps(strategy.get("src_feature_cols", [])),
        fe_feature_cols=json.dumps(strategy.get("fe_feature_cols", [])),
        numerical_cols=json.dumps(strategy.get("numerical_cols", [])),
        categorical_cols=json.dumps(strategy.get("categorical_cols", [])),
        feature_engineering=_normalize_fe(strategy.get("feature_engineering", "")),
        param_search_space=search_space,
        model_import=model_import,
        model_class=model_class,
        model_init_tune=model_init_tune,
        model_init_best=model_init_best,
        submission_dir=str(Path(state.get("submission_dir", ""))),
        submission_filename=sub_name,
        train_file=_find_csv(profile, state.get("data_dir", ""), "train"),
        test_file=_find_csv(profile, state.get("data_dir", ""), "test"),
    )

    out = Path(script_dir) / f"tune_{model_name.lower().replace(' ', '_')}.py"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(content, encoding="utf-8")
    _log(f"Saved tuning script → {out}")
    return out


def _normalize_fe(code: str) -> str:
    """Normalize LLM feature engineering code for template insertion."""
    from kagglemate.graph.nodes.baseline_node import _normalize_indent
    return _normalize_indent(code)


def _find_csv(profile: dict, data_dir: str, pattern: str) -> str:
    d = Path(data_dir)
    if d.exists():
        for f in d.glob("*.csv"):
            if pattern.lower() in f.name.lower():
                return f.name
    return f"{pattern}.csv"


def _log(msg: str):
    print(f"  [tune] {msg}")
