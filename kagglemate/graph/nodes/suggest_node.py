"""Suggest Node — generates next_steps.md from experiment history + research.

This is where the agent "thinks" about what to do next. It feeds:
- All experiment history (CV scores, models, features)
- Research findings from public notebooks
- Current best scores (CV + LB)
- Any CV/LB gap
...to DeepSeek V4 Pro, which returns a prioritized list of next experiments.

The output is actionable: each suggestion names a specific experiment,
explains why it's promising, and estimates expected improvement.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.llm_client import simple_prompt
from kagglemate.memory.experiment_store import ExperimentStore


SUGGEST_PROMPT = """You are a Kaggle grandmaster strategist. Based on the following information, recommend the next 3-5 experiments that would most likely improve the competition score.

## Competition
- Name: {competition_name}
- Type: {competition_type}
- Metric: {evaluation_metric}

## Current Status
- Best CV Score: {best_cv}
- Best LB Score: {best_lb}
- CV/LB Gap: {cv_lb_gap}
- Total experiments run: {total_experiments}

## Experiment History (newest first)
{experiment_table}

## Research Findings (from top public notebooks)
{research_summary}

## What Has Been Tried
{tried_summary}

## What Has NOT Been Tried (from research)
{untried_techniques}

## Instructions
1. Diagnose the current situation in 1-2 sentences
2. Recommend 3-5 specific, actionable experiments
3. For each experiment, specify:
   - name: a short slug like "target_encode_catboost"
   - what_to_do: 2-3 sentences explaining the change
   - expected_impact: "high" / "medium" / "low"
   - reason: why this is promising based on the research or experimentation history
   - cv_improvement_estimate: approximate score improvement (e.g. "+0.005", "+0.01")
   - risk: what could go wrong
4. Prioritize by expected impact: high first, then medium, then low
5. Be SPECIFIC — say exactly which model, which features, which params to change
6. If the CV/LB gap is large (>0.03), prioritize reducing overfitting
7. If no public notebooks were found, rely on your own Kaggle experience
8. Never suggest "try different things" — name exact techniques

Output as JSON:
```json
{{
  "situation": "One sentence diagnosis.",
  "recommendations": [
    {{
      "name": "experiment_slug",
      "what_to_do": "...",
      "expected_impact": "high",
      "reason": "...",
      "cv_improvement_estimate": "+0.005",
      "risk": "..."
    }}
  ]
}}
```"""


def run(state: KaggleAgentState) -> dict:
    """Generate next-steps recommendations.

    Reads experiment history and research findings, calls DeepSeek V4 Pro,
    and writes next_steps.md to the reports directory.
    """
    slug = state["competition_slug"]
    _log(f"Generating next-step suggestions for {slug}")

    store = ExperimentStore(slug)
    all_exps = store.list_all()
    best_exp = store.get_best()

    best_cv = (best_exp.get("cv_score") or 0.0) if best_exp else 0.0
    best_lb = (best_exp.get("lb_score") if best_exp else None) or "N/A"

    # CV/LB gap
    cv_lb_gap = store.cv_lb_gap()
    gap_str = f"{cv_lb_gap:.5f}" if cv_lb_gap is not None else "N/A (no LB score yet)"

    # Build experiment history table
    exp_table = _build_experiment_table(all_exps)

    # Build "what has been tried" summary
    tried = _build_tried_summary(all_exps)

    # Read research summary
    research_text = _read_research_summary(state)

    # Build "what hasn't been tried" from research
    untried = _build_untried(all_exps, state)

    # ── Call LLM ──
    suggestions = _get_suggestions(
        competition_name=state.get("competition_name", slug),
        competition_type=state.get("competition_type", "unknown"),
        evaluation_metric=state.get("evaluation_metric", "unknown"),
        best_cv=f"{best_cv:.5f}" if best_cv > 0 else "N/A",
        best_lb=str(best_lb),
        cv_lb_gap=gap_str,
        total_experiments=len(all_exps),
        experiment_table=exp_table,
        research_summary=research_text,
        tried_summary=tried,
        untried_techniques=untried,
    )

    # ── Save next_steps.md ──
    report_dir = state.get("report_dir", "")
    if report_dir:
        _save_next_steps(Path(report_dir), state, suggestions, all_exps)

    return {
        "current_phase": "suggest",
    }


def _get_suggestions(**kwargs) -> dict:
    """Get LLM recommendations, with fallback."""
    prompt = SUGGEST_PROMPT.format(**kwargs)

    try:
        raw = simple_prompt(prompt)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        return json.loads(raw.strip())
    except Exception as e:
        _log(f"LLM suggest failed: {e}")
        return {
            "situation": "Unable to generate AI suggestions (LLM error).",
            "recommendations": [
                {
                    "name": "target_encoding",
                    "what_to_do": "Add target encoding for high-cardinality categorical features using leave-one-out or KFold scheme.",
                    "expected_impact": "medium",
                    "reason": "Target encoding consistently improves tree-based models on tabular data.",
                    "cv_improvement_estimate": "+0.003",
                    "risk": "Overfitting if not done with proper CV.",
                },
                {
                    "name": "hyperparameter_tuning",
                    "what_to_do": "Run Optuna hyperparameter optimization: tune learning_rate, num_leaves, feature_fraction, lambda_l1/l2.",
                    "expected_impact": "medium",
                    "reason": "Default params are rarely optimal.",
                    "cv_improvement_estimate": "+0.002",
                    "risk": "Time-consuming. Risk of overfitting to CV.",
                },
                {
                    "name": "ensemble",
                    "what_to_do": "Blend LightGBM + CatBoost predictions using simple average.",
                    "expected_impact": "medium",
                    "reason": "Different tree architectures capture different patterns.",
                    "cv_improvement_estimate": "+0.003",
                    "risk": "Requires running both models. May overfit if CV scores are correlated.",
                },
            ],
        }


def _build_experiment_table(exps: list[dict]) -> str:
    """Format experiments as a table for the LLM prompt."""
    if not exps:
        return "(No experiments yet)"
    lines = ["| # | Name | Model | CV | LB | Status |",
             "|---|------|-------|----|----|--------|"]
    for e in exps[:15]:
        lines.append(
            f"| {e['id']} | {e.get('experiment_name', '?')[:25]} | "
            f"{e.get('model_name', '?')[:12]} | "
            f"{e.get('cv_score', 'N/A')} | {e.get('lb_score', 'N/A')} | "
            f"{e.get('status', '?')} |"
        )
    return "\n".join(lines)


def _build_tried_summary(exps: list[dict]) -> str:
    """Summarize what's been tried."""
    if not exps:
        return "Nothing tried yet — this is the first experiment."

    models = set()
    features_used = set()
    for e in exps:
        if e.get("model_name"):
            models.add(e["model_name"])
        for f in (e.get("features") or []):
            features_used.add(f)

    parts = []
    if models:
        parts.append(f"Models: {', '.join(sorted(models))}")
    if features_used:
        parts.append(f"Features used across experiments: {len(features_used)} unique columns")
    if not parts:
        parts.append("Basic baseline only.")
    return "\n".join(parts)


def _build_untried(exps: list[dict], state: KaggleAgentState) -> str:
    """Build a list of techniques from research that haven't been tried."""
    notebooks = state.get("notebook_summaries", [])
    if not notebooks:
        return "No public notebook research available for comparison."

    tried_models = {e.get("model_name", "").lower() for e in exps}
    all_notebook_techniques: list[str] = []
    for nb in notebooks:
        all_notebook_techniques.extend(nb.get("key_techniques", []))

    # Deduplicate
    seen = set()
    unique_techniques = []
    for t in all_notebook_techniques:
        if t.lower() not in seen:
            seen.add(t.lower())
            unique_techniques.append(t)

    # Filter out things clearly tried
    untried = []
    for t in unique_techniques:
        t_lower = t.lower()
        # Simple heuristic: if technique name contains a tried model, skip
        if any(m in t_lower for m in tried_models):
            continue
        untried.append(t)

    if not untried:
        return "Most techniques from public notebooks have been attempted."
    return "\n".join(f"- {t}" for t in untried[:10])


def _read_research_summary(state: KaggleAgentState) -> str:
    """Read the research summary file if it exists."""
    report_dir = state.get("report_dir", "")
    if report_dir:
        path = Path(report_dir) / "research_summary.md"
        if path.exists():
            return path.read_text()[:3000]  # first 3000 chars
    return "No research summary available yet."


def _save_next_steps(
    report_dir: Path,
    state: KaggleAgentState,
    suggestions: dict,
    all_exps: list[dict],
) -> Path:
    """Render next_steps.md from template + LLM suggestions."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent.parent.parent / "templates"))
    )
    template = env.get_template("next_steps_template.md")

    # Find best experiment
    best_exp = all_exps[0] if all_exps else None
    best_cv = best_exp.get("cv_score", "N/A") if best_exp else "N/A"
    best_lb = best_exp.get("lb_score", "N/A") if best_exp else "N/A"

    content = template.render(
        competition_name=state.get("competition_name", state["competition_slug"]),
        competition_slug=state["competition_slug"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        situation=suggestions.get("situation", "No diagnosis available."),
        best_cv=best_cv,
        best_lb=best_lb,
        total_experiments=len(all_exps),
        recommendations=suggestions.get("recommendations", []),
    )

    out = report_dir / "next_steps.md"
    out.write_text(content, encoding="utf-8")
    _log(f"Saved next_steps.md → {out}")
    return out


def _log(msg: str):
    print(f"  [suggest] {msg}")
