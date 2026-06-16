"""Research Node — public notebook discovery and summarization.

1. Fetches the top-N kernels from Kaggle CLI (by votes)
2. Batches kernel metadata to DeepSeek for technique extraction
3. Identifies common patterns and recommends a baseline approach
4. Produces research_summary.md
"""

from __future__ import annotations

import json
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState, NotebookSummary
from kagglemate.config import config
from kagglemate.tools.kaggle_cli import KaggleCLI
from kagglemate.tools.llm_client import simple_prompt


# ── LLM prompts ──

NOTEBOOK_BATCH_PROMPT = """You are analyzing Kaggle competition notebooks. Below is a list of public notebooks (title, author, votes). For each one, infer the likely approach based on the title and any available metadata.

For each notebook, output a JSON object with these fields:
- ref: the kernel ref string
- title: notebook title
- author: author name
- votes: vote count
- model: likely model used (e.g. "LightGBM", "XGBoost", "CatBoost", "Ensemble", "Neural Network", "Unknown")
- cv_method: likely CV strategy (e.g. "StratifiedKFold", "KFold", "GroupKFold", "Unknown")
- lb_score: null (we'll get this from submission data later)
- key_techniques: list of 2-4 key techniques likely used (from title hints)
- worth_reproducing: true if it sounds like a high-quality approach, false otherwise
- notes: one sentence about what makes this notebook notable

## Notebook list
{notebook_list}

## Competition context
Competition type: {competition_type}
Evaluation metric: {evaluation_metric}

Output as a JSON array. Only output the JSON, no explanation.
```json
[{{"ref": "...", "title": "...", ...}}, ...]
```"""


PATTERN_ANALYSIS_PROMPT = """Based on the following notebook summaries from a Kaggle competition, identify the top 3-5 common patterns and recommend a baseline approach.

## Competition
- Type: {competition_type}
- Metric: {evaluation_metric}

## Notebook Summaries
{summaries_json}

## Instructions
1. Identify the most common model(s) used
2. Identify the most common CV strategy
3. Identify the most common feature engineering techniques
4. Recommend a baseline: what model, CV, and feature engineering to start with
5. List 3-5 high-ROI improvement ideas beyond the baseline

Output as JSON:
```json
{{
  "common_patterns": ["pattern 1", "pattern 2", ...],
  "recommended_model": "LightGBM",
  "recommended_cv": "StratifiedKFold(n=5)",
  "recommended_fe": "Label encoding for categoricals, no feature engineering for baseline",
  "improvement_ideas": ["idea 1", "idea 2", ...],
  "notebooks_to_study": [{{"ref": "...", "reason": "..."}}, ...]
}}
```"""


def run(state: KaggleAgentState) -> dict:
    """Discover and summarize public notebooks for a competition.

    Returns state updates with notebook_summaries and research_complete flag.
    Writes research_summary.md to the reports directory.
    """
    slug = state["competition_slug"]
    _log(f"Researching notebooks for: {slug}")

    # ── Step 1: Fetch kernel list ──
    kernels = KaggleCLI.list_kernels(slug, sort_by="votes", limit=config.MAX_RESEARCH_NOTEBOOKS)
    _log(f"Fetched {len(kernels)} kernels from Kaggle")

    if not kernels:
        return {
            "notebook_summaries": [],
            "research_complete": True,
            "current_phase": "research",
            "errors": [f"No public notebooks found for {slug} (new competition?)"],
        }

    # ── Step 2: Batch summarize with LLM ──
    notebook_list_json = json.dumps([
        {
            "ref": k.get("ref", ""),
            "title": k.get("title", ""),
            "author": k.get("author", ""),
            "votes": int(k.get("totalVotes", 0) or 0),
        }
        for k in kernels
    ], indent=2, ensure_ascii=False)

    summaries: list[NotebookSummary] = []
    try:
        raw = simple_prompt(
            NOTEBOOK_BATCH_PROMPT.format(
                notebook_list=notebook_list_json,
                competition_type=state.get("competition_type", "unknown"),
                evaluation_metric=state.get("evaluation_metric", "unknown"),
            ),
            use_flash=True,  # bulk summarization — use cheaper model
        )
        # Parse JSON array from response
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        parsed = json.loads(raw.strip())
        summaries = [
            {
                "ref": s.get("ref", ""),
                "title": s.get("title", ""),
                "author": s.get("author", ""),
                "votes": s.get("votes", 0),
                "model": s.get("model", "Unknown"),
                "cv_method": s.get("cv_method", "Unknown"),
                "lb_score": s.get("lb_score"),
                "key_techniques": s.get("key_techniques", []),
                "worth_reproducing": s.get("worth_reproducing", False),
                "notes": s.get("notes", ""),
            }
            for s in parsed
        ]
    except Exception as e:
        _log(f"LLM summarization failed: {e}. Falling back to raw metadata.")
        summaries = [
            {
                "ref": k.get("ref", ""),
                "title": k.get("title", ""),
                "author": k.get("author", ""),
                "votes": int(k.get("totalVotes", 0) or 0),
                "model": "Unknown",
                "cv_method": "Unknown",
                "lb_score": None,
                "key_techniques": [],
                "worth_reproducing": False,
                "notes": f"LLM analysis failed; manual review needed.",
            }
            for k in kernels
        ]

    # ── Step 3: Pattern analysis ──
    pattern_analysis = {
        "common_patterns": [],
        "recommended_model": "LightGBM",
        "recommended_cv": "StratifiedKFold(n=5)",
        "recommended_fe": "Label encoding, no advanced FE",
        "improvement_ideas": [],
        "notebooks_to_study": [],
    }

    try:
        raw = simple_prompt(
            PATTERN_ANALYSIS_PROMPT.format(
                competition_type=state.get("competition_type", "unknown"),
                evaluation_metric=state.get("evaluation_metric", "unknown"),
                summaries_json=json.dumps(summaries, indent=2, ensure_ascii=False),
            )
        )
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        pattern_analysis = json.loads(raw.strip())
    except Exception as e:
        _log(f"Pattern analysis failed: {e}")

    # ── Step 4: Save research_summary.md ──
    report_dir = state.get("report_dir", "")
    if report_dir:
        _save_research_markdown(
            Path(report_dir),
            state,
            summaries,
            pattern_analysis,
        )

    return {
        "notebook_summaries": summaries,
        "research_complete": True,
        "current_phase": "research",
    }


# ── Helpers ──

def _save_research_markdown(
    report_dir: Path,
    state: KaggleAgentState,
    summaries: list[NotebookSummary],
    patterns: dict,
) -> Path:
    """Render research_summary.md from template data."""
    from jinja2 import Environment, FileSystemLoader

    env = Environment(
        loader=FileSystemLoader(str(Path(__file__).parent.parent.parent / "templates"))
    )
    template = env.get_template("research_summary_template.md")

    content = template.render(
        competition_name=state.get("competition_name", state["competition_slug"]),
        competition_slug=state["competition_slug"],
        generated_at=datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC"),
        notebook_count=len(summaries),
        notebooks=summaries,
        common_patterns="\n".join(
            f"- {p}" for p in patterns.get("common_patterns", [])
        ),
        recommended_model=patterns.get("recommended_model", "LightGBM"),
        recommended_cv=patterns.get("recommended_cv", "StratifiedKFold(n=5)"),
        recommended_fe=patterns.get("recommended_fe", "Baseline — no advanced FE"),
        improvement_ideas=patterns.get("improvement_ideas", []),
        notebooks_to_study=patterns.get("notebooks_to_study", []),
    )

    out = report_dir / "research_summary.md"
    out.write_text(content, encoding="utf-8")
    _log(f"Saved research_summary.md → {out}")
    return out


def _log(msg: str):
    print(f"  [research] {msg}")
