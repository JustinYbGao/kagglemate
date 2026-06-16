"""Analyze Node — data profiling and task type identification.

Reads CSV files from the competition data directory, runs DataProfiler
for statistical analysis, then uses the LLM to determine the competition
type (classification vs regression) and evaluation metric.

Outputs: data_profile.md
"""

from __future__ import annotations

import json
from pathlib import Path

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.data_profiler import DataProfiler
from kagglemate.tools.llm_client import simple_prompt


# ── LLM prompt: determine task type from data profile ──

TASK_TYPE_PROMPT = """You are analyzing a Kaggle competition dataset. Based on the data profile below, determine:

1. **competition_type**: one of "tabular_classification", "tabular_regression", "image", "text", "time_series", "other"
2. **evaluation_metric**: the most likely evaluation metric (e.g. "auc", "accuracy", "rmse", "logloss", "f1", "mae", "r2")
3. **reasoning**: one sentence explaining your decision

## Data Profile

```json
{profile_json}
```

## Submission Format

Sample submission columns: {submission_cols}

## Rules for Classification vs Regression

- If the target column has <= 20 unique values → classification
- If the target column is a float with many unique values → regression
- If sample_submission has a probability-like column (values 0-1) → binary classification (metric: auc)
- If sample_submission has integer class labels → classification (metric: accuracy or logloss)
- If the competition name contains "regression" or the metric is clearly RMSE/MAE → regression

Output ONLY valid JSON with keys: competition_type, evaluation_metric, reasoning
```json
{{"competition_type": "...", "evaluation_metric": "...", "reasoning": "..."}}
```"""


def run(state: KaggleAgentState) -> dict:
    """Profile the competition data and identify the task type.

    Returns state updates including data_profile, competition_type,
    and evaluation_metric.
    """
    data_dir = state.get("data_dir", "")
    if not data_dir:
        return {"errors": ["No data_dir in state. Did init_node run?"], "current_phase": "analyze"}

    # ── Step 1: Statistical profiling (no LLM) ──
    profiler = DataProfiler(data_dir)
    profile = profiler.run()

    report_dir = state.get("report_dir", "")
    if report_dir:
        profiler.save_markdown(profile, Path(report_dir) / "data_profile.md")

    # ── Step 2: LLM determines task type ──
    profile_json = json.dumps({
        "train_rows": profile.get("train_rows"),
        "test_rows": profile.get("test_rows"),
        "numerical_cols": profile.get("numerical_cols"),
        "categorical_cols": profile.get("categorical_cols"),
        "target_col": profile.get("target_col"),
        "target_distribution": profile.get("target_distribution"),
        "column_names": profile.get("columns"),
        "missing_values": profile.get("missing_values"),
    }, indent=2, ensure_ascii=False)

    competition_type = "tabular_classification"  # fallback
    evaluation_metric = "unknown"

    try:
        raw = simple_prompt(
            TASK_TYPE_PROMPT.format(
                profile_json=profile_json,
                submission_cols=json.dumps(profile.get("submission_cols", [])),
            )
        )
        # Extract JSON from response (may be wrapped in ```json)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        parsed = json.loads(raw.strip())
        competition_type = parsed.get("competition_type", competition_type)
        evaluation_metric = parsed.get("evaluation_metric", "unknown")
    except Exception:
        # Fallback: use heuristics
        if profile.get("target_distribution"):
            n_classes = len(profile["target_distribution"].split(","))
            if n_classes <= 20:
                competition_type = "tabular_classification"
                evaluation_metric = "auc" if n_classes == 2 else "logloss"
            else:
                competition_type = "tabular_regression"
                evaluation_metric = "rmse"
        else:
            competition_type = "tabular_regression"
            evaluation_metric = "rmse"

    return {
        "data_profile": profile,
        "competition_type": competition_type,
        "evaluation_metric": evaluation_metric,
        "submission_cols_known": bool(profile.get("submission_cols")),
        "current_phase": "analyze",
    }
