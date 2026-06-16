"""Plan Node — generates SPEC.md and rules_checklist.md.

Takes all previous analysis (data profile, notebook research, task type)
and produces the strategy documents that guide the rest of the competition.
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.llm_client import simple_prompt


def run(state: KaggleAgentState) -> dict:
    """Generate SPEC.md and rules_checklist.md from research findings.

    Returns state updates with paths to generated documents.
    """
    report_dir = state.get("report_dir", "")
    if not report_dir:
        return {"errors": ["No report_dir in state"], "current_phase": "plan"}

    report_path = Path(report_dir)
    slug = state["competition_slug"]

    _log(f"Generating SPEC.md and rules_checklist.md for {slug}")

    # ── SPEC.md ──
    spec_path = _generate_spec(report_path, state)

    # ── rules_checklist.md ──
    rules_path = _generate_rules_checklist(report_path, state)

    return {
        "spec_path": str(spec_path),
        "research_summary_path": str(report_path / "research_summary.md"),
        "rules_checklist_path": str(rules_path),
        "current_phase": "plan",
    }


def _generate_spec(report_dir: Path, state: KaggleAgentState) -> Path:
    """Render SPEC.md using the Jinja2 template + LLM-generated content."""

    profile = state.get("data_profile") or {}
    summaries = state.get("notebook_summaries", [])

    # ── Use LLM to generate the narrative sections ──
    prompt = f"""Based on the following information about a Kaggle competition, generate the narrative sections for a SPEC.md document.

## Competition Info
- Name: {state.get("competition_name", "")}
- Type: {state.get("competition_type", "unknown")}
- Metric: {state.get("evaluation_metric", "unknown")}
- Train rows: {profile.get("train_rows", "?")}
- Test rows: {profile.get("test_rows", "?")}
- Target: {profile.get("target_col", "?")}

## Top Notebooks Summary
{json.dumps(summaries[:10], indent=2, ensure_ascii=False)}

Generate these sections:
1. **notebook_findings** — A 2-3 paragraph summary of what the top public notebooks are doing
2. **common_patterns** — Bullet list of 3-5 patterns seen across notebooks
3. **baseline_recommendation** — One paragraph: what model, CV method, and feature engineering to start with
4. **high_roi_improvements** — List of 3-5 improvements with specific expected gains
5. **baseline_model** — Just the model name (e.g. "LightGBM")
6. **cv_strategy** — Just the CV method (e.g. "StratifiedKFold(n=5)")
7. **baseline_target** — A realistic CV score target for the baseline

Output as JSON:
```json
{{
  "notebook_findings": "...",
  "common_patterns": "...",
  "baseline_recommendation": "...",
  "high_roi_improvements": ["...", ...],
  "baseline_model": "LightGBM",
  "cv_strategy": "StratifiedKFold(n=5)",
  "baseline_target": "0.xx"
}}
```"""

    llm_sections = {}
    try:
        raw = simple_prompt(prompt)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        llm_sections = json.loads(raw.strip())
    except Exception as e:
        _log(f"LLM sections failed: {e}, using defaults")
        llm_sections = {
            "notebook_findings": "See research_summary.md for details.",
            "common_patterns": "- Tree-based models dominate\n- KFold CV is standard",
            "baseline_recommendation": "Start with LightGBM using all numerical features.",
            "high_roi_improvements": [
                "Add target encoding for categorical features",
                "Try CatBoost as an alternative model",
                "Hyperparameter tuning with Optuna",
                "Feature engineering based on domain knowledge",
                "Simple ensemble of LightGBM + CatBoost",
            ],
            "baseline_model": "LightGBM",
            "cv_strategy": "StratifiedKFold(n=5)",
            "baseline_target": "TBD",
        }

    # ── Render template ──
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent.parent.parent / "templates"))
    )
    template = env.get_template("spec_template.md")

    submission_cols = profile.get("submission_cols", [])
    numerical = profile.get("numerical_cols", [])
    categorical = profile.get("categorical_cols", [])

    content = template.render(
        competition_name=state.get("competition_name", state["competition_slug"]),
        competition_slug=state["competition_slug"],
        competition_type=state.get("competition_type", "unknown"),
        evaluation_metric=state.get("evaluation_metric", "unknown"),
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        train_rows=profile.get("train_rows", "?"),
        test_rows=profile.get("test_rows", "?"),
        feature_count=len(profile.get("columns", [])),
        target_col=profile.get("target_col", "?"),
        submission_cols=", ".join(submission_cols),
        submission_rows=profile.get("submission_rows", "?"),
        numerical_count=len(numerical),
        categorical_count=len(categorical),
        numerical_cols=", ".join([f"`{c}`" for c in numerical[:15]]),
        categorical_cols=", ".join([f"`{c}`" for c in categorical[:15]]),
        target_distribution=profile.get("target_distribution", ""),
        missing_values=profile.get("missing_values", {}),
        # LLM-generated sections
        notebook_findings=llm_sections.get("notebook_findings", ""),
        common_patterns=llm_sections.get("common_patterns", ""),
        baseline_recommendation=llm_sections.get("baseline_recommendation", ""),
        high_roi_improvements=llm_sections.get("high_roi_improvements", []),
        baseline_model=llm_sections.get("baseline_model", "LightGBM"),
        cv_strategy=llm_sections.get("cv_strategy", "StratifiedKFold(n=5)"),
        baseline_target=llm_sections.get("baseline_target", "TBD"),
        internet_note=_internet_note(state),
        external_data_note=_external_data_note(state),
    )

    out = report_dir / "SPEC.md"
    out.write_text(content, encoding="utf-8")
    _log(f"Saved SPEC.md → {out}")
    return out


def _generate_rules_checklist(report_dir: Path, state: KaggleAgentState) -> Path:
    """Render rules_checklist.md from template."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent.parent.parent / "templates"))
    )
    template = env.get_template("rules_checklist_template.md")

    profile = state.get("data_profile") or {}

    content = template.render(
        competition_name=state.get("competition_name", state["competition_slug"]),
        competition_slug=state["competition_slug"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        submission_cols=", ".join(profile.get("submission_cols", [])),
        submission_rows=profile.get("submission_rows", "?"),
        file_type_note="Standard CSV (unless competition specifies .zip)",
        daily_limit_note="Check competition rules page (typically 5/day for Kaggle)",
        team_size_note="Check competition rules page",
        external_data_note=_external_data_note(state),
        internet_note=_internet_note(state),
        pretrained_note="Check competition rules page",
        gpu_note="Not required for tabular ML (required for deep learning approaches)",
        code_competition_note="Check competition type on Kaggle page",
        late_submission_note="Check competition deadline on Kaggle page",
    )

    out = report_dir / "rules_checklist.md"
    out.write_text(content, encoding="utf-8")
    _log(f"Saved rules_checklist.md → {out}")
    return out


def _internet_note(state: KaggleAgentState) -> str:
    """Heuristic: internet is usually disabled in Code Competitions."""
    ctype = state.get("competition_type", "")
    if "code" in ctype.lower():
        return "Likely DISABLED (Code Competition). Verify on rules page."
    return "Likely allowed for notebook competitions. Verify on rules page."


def _external_data_note(state: KaggleAgentState) -> str:
    return "Check competition rules page. Most competitions prohibit external data."


def _log(msg: str):
    print(f"  [plan] {msg}")
