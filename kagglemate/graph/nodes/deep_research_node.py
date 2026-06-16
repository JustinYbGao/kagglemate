"""Deep Research Node — multi-source synthesis across Kaggle, arXiv, and Web.

Flow:
1. Research Kaggle notebooks (existing pipeline)
2. Extract technical keywords from notebook findings
3. Search arXiv for related papers
4. Search web for discussions/blogs
5. Cross-source synthesis: identify consensus, contradictions, novel approaches
6. Generate deep_research.md with full citations
"""

from __future__ import annotations

import concurrent.futures
from datetime import datetime, timezone
from pathlib import Path

from kagglemate.graph.state import KaggleAgentState
from kagglemate.research_providers import (
    KaggleProvider, ArxivProvider, WebProvider, SourceItem,
)
from kagglemate.tools.llm_client import simple_prompt


# ── Synthesis prompts ──

KEYWORD_EXTRACTION_PROMPT = """Based on the following Kaggle competition context, extract 3-5 technical search queries for finding relevant academic papers and web resources.

## Competition
- Name: {competition_name}
- Type: {competition_type}
- Metric: {evaluation_metric}

## Top Notebook Techniques
{notebook_techniques}

Generate 3-5 search queries. Each should be:
- Short (2-6 words)
- Focused on a specific technique or method
- Suitable for arXiv or Google search
- In English

Output as JSON array of strings:
["query 1", "query 2", "query 3"]"""


SYNTHESIS_PROMPT = """You are a Kaggle Grandmaster research strategist. Synthesize findings from multiple sources into an actionable deep research report.

## Competition Context
- Name: {competition_name}
- Type: {competition_type}
- Metric: {evaluation_metric}

## Source 1: Kaggle Notebooks ({n_kaggle} analyzed)
{kaggle_findings}

## Source 2: arXiv Papers ({n_arxiv} found)
{arxiv_findings}

## Source 3: Web Search ({n_web} found)
{web_findings}

## Instructions

Analyze ALL sources and produce a structured report with these sections:

1. **Executive Summary / 执行摘要** (3-4 sentences): The single most important takeaway across all sources.

2. **Method Matrix / 方法矩阵**: Table comparing techniques across sources.
   Format: | Technique | Kaggle | arXiv | Web | Consensus? |
   If a technique appears in multiple sources, mark Consensus=Yes.

3. **Consensus Techniques / 共识方法** (3-5 items): What techniques do multiple independent sources agree on? These are the SAFEST bets.

4. **Novel / Contested Approaches / 前沿/争议方法** (2-4 items): Techniques that only appear in one source, or where sources disagree. Higher risk, potentially higher reward.

5. **Paper-to-Practice Bridge / 论文落地建议** (2-3 items): Specific academic papers whose methods could be adapted for this competition. Include: what the paper does, how to adapt it, expected difficulty.

6. **Recommended Strategy / 推荐策略**:
   - Immediate actions / 立即执行 (next 24h)
   - Short-term experiments / 短期实验 (3-7 days)
   - Long-term exploration / 长期探索 (if time permits)

7. **Key References / 关键引用** (numbered list): The 10 most important items across all sources, with why each matters.

Output the report in Markdown. Use Chinese+English bilingual headers where helpful (e.g. "## 3. Consensus Techniques / 共识方法").
Be SPECIFIC — name exact models, parameters, papers. No vague suggestions."""


def run(state: KaggleAgentState) -> dict:
    """Execute deep multi-source research.

    Requires: competition_slug, competition_name, competition_type,
              evaluation_metric, and ideally notebook_summaries from prior research.
    """
    slug = state["competition_slug"]
    _log(f"Deep research: {slug}")

    # ── Phase 1: Kaggle notebooks (parallel with other prep) ──
    kaggle = KaggleProvider()
    _log("Searching Kaggle notebooks...")
    kaggle_items = kaggle.search(slug, max_results=20)
    _log(f"  Found {len(kaggle_items)} notebooks")

    # ── Phase 2: Extract technique keywords ──
    notebook_techniques = _extract_technique_text(state, kaggle_items)
    keywords = _extract_keywords(state, notebook_techniques)
    _log(f"Extracted keywords: {keywords}")

    # ── Phase 3: Parallel search arXiv + Web ──
    arxiv = ArxivProvider()
    web = WebProvider()

    arxiv_items: list[SourceItem] = []
    web_items: list[SourceItem] = []

    # Search each keyword across both providers
    all_arxiv: list[SourceItem] = []
    all_web: list[SourceItem] = []

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futures = {}
        for kw in keywords[:4]:  # limit to 4 keywords to control API load
            futures[pool.submit(arxiv.search, kw, 5)] = ("arxiv", kw)
            futures[pool.submit(web.search, kw, 5)] = ("web", kw)

        for future in concurrent.futures.as_completed(futures):
            source, kw = futures[future]
            try:
                items = future.result(timeout=30)
                if source == "arxiv":
                    all_arxiv.extend(items)
                    _log(f"  arXiv '{kw}': {len(items)} results")
                else:
                    all_web.extend(items)
                    _log(f"  Web '{kw}': {len(items)} results")
            except Exception:
                pass

    # Deduplicate
    arxiv_items = _dedup_by_url(all_arxiv)[:15]
    web_items = _dedup_by_url(all_web)[:15]

    _log(f"Total: {len(kaggle_items)} Kaggle + {len(arxiv_items)} arXiv + {len(web_items)} Web")

    # ── Phase 4: Synthesis ──
    _log("Synthesizing findings...")
    report_content = _synthesize(state, kaggle_items, arxiv_items, web_items)

    # ── Phase 5: Save report ──
    report_dir = state.get("report_dir", "")
    if report_dir:
        out = Path(report_dir) / "deep_research.md"
        out.write_text(report_content, encoding="utf-8")
        _log(f"Saved → {out}")

    return {
        "current_phase": "research",
    }


def _extract_technique_text(state: KaggleAgentState, items: list[SourceItem]) -> str:
    """Build a technique summary from notebook findings."""
    nb_summaries = state.get("notebook_summaries", [])
    if nb_summaries:
        lines = []
        for nb in nb_summaries[:10]:
            techniques = ", ".join(nb.get("key_techniques", []))
            lines.append(f"- {nb.get('title', '')}: {nb.get('model', '')} — {techniques}")
        return "\n".join(lines)

    # Fallback: from SourceItem snippets
    return "\n".join(
        f"- {item.title}: {item.snippet[:120]}"
        for item in items[:10]
    )


def _extract_keywords(state: KaggleAgentState, technique_text: str) -> list[str]:
    """Use LLM to extract technical search keywords."""
    prompt = KEYWORD_EXTRACTION_PROMPT.format(
        competition_name=state.get("competition_name", state["competition_slug"]),
        competition_type=state.get("competition_type", "unknown"),
        evaluation_metric=state.get("evaluation_metric", "unknown"),
        notebook_techniques=technique_text[:3000],
    )

    try:
        import json
        raw = simple_prompt(prompt, use_flash=True)
        if "```json" in raw:
            raw = raw.split("```json")[1].split("```")[0]
        elif "```" in raw:
            raw = raw.split("```")[1].split("```")[0]
        keywords = json.loads(raw.strip())
        return keywords if isinstance(keywords, list) else []
    except Exception:
        # Fallback: generic queries based on competition type
        comp_type = state.get("competition_type", "")
        if "classif" in comp_type:
            return ["gradient boosting feature engineering", "ensemble methods kaggle",
                    "tabular data deep learning"]
        elif "regression" in comp_type:
            return ["gradient boosting regression", "feature engineering regression",
                    "tabular regression ensemble"]
        else:
            slug = state["competition_slug"]
            return [f"{slug} kaggle solution", f"{slug} machine learning",
                    "competition machine learning technique"]


def _synthesize(state: KaggleAgentState,
                kaggle: list[SourceItem], arxiv: list[SourceItem],
                web: list[SourceItem]) -> str:
    """LLM synthesis of all sources into a deep research report."""

    # Format sources as text (truncate for token budget)
    def fmt_items(items: list[SourceItem], max_items: int = 15) -> str:
        return "\n\n".join(
            item.to_markdown() for item in items[:max_items]
        ) or "(No results / 无结果)"

    kaggle_text = fmt_items(kaggle)
    arxiv_text = fmt_items(arxiv, 10)
    web_text = fmt_items(web, 10)

    prompt = SYNTHESIS_PROMPT.format(
        competition_name=state.get("competition_name", state["competition_slug"]),
        competition_type=state.get("competition_type", "unknown"),
        evaluation_metric=state.get("evaluation_metric", "unknown"),
        n_kaggle=len(kaggle),
        n_arxiv=len(arxiv),
        n_web=len(web),
        kaggle_findings=kaggle_text[:4000],
        arxiv_findings=arxiv_text[:3000],
        web_findings=web_text[:3000],
    )

    try:
        return simple_prompt(prompt)
    except Exception:
        return _fallback_report(state, kaggle, arxiv, web)


def _fallback_report(state: KaggleAgentState,
                     kaggle: list[SourceItem], arxiv: list[SourceItem],
                     web: list[SourceItem]) -> str:
    """Generate a basic report without LLM."""
    parts = [
        f"# Deep Research Report / 深度调研报告 — {state.get('competition_name', state['competition_slug'])}",
        f"\n**Generated / 生成时间**: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}",
        f"\n**Sources / 数据源**: {len(kaggle)} Kaggle, {len(arxiv)} arXiv, {len(web)} Web\n",
        "\n## Kaggle Notebooks / Kaggle 高分方案\n",
    ]
    for item in kaggle[:15]:
        parts.append(item.to_markdown())

    if arxiv:
        parts.append("\n## arXiv Papers / 学术论文\n")
        for item in arxiv[:10]:
            parts.append(item.to_markdown())

    if web:
        parts.append("\n## Web Resources / 网络资源\n")
        for item in web[:10]:
            parts.append(item.to_markdown())

    parts.append("\n## Note / 备注\n")
    parts.append("LLM synthesis unavailable. Manual review recommended. / LLM 合成失败，建议手动审阅。")

    return "\n".join(parts)


def _dedup_by_url(items: list[SourceItem]) -> list[SourceItem]:
    """Remove duplicate items by URL."""
    seen = set()
    result = []
    for item in items:
        if item.url not in seen:
            seen.add(item.url)
            result.append(item)
    return result


def _log(msg: str):
    print(f"  [deep-research] {msg}")
