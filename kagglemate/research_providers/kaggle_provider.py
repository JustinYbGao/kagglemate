"""Kaggle Notebook Provider — searches public kernels for a competition.

Wraps the existing KaggleCLI.list_kernels + DeepSeek summarization.
This is the original research pipeline, now refactored as a provider.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

from kagglemate.research_providers.base import ResearchProvider, SourceItem
from kagglemate.tools.kaggle_cli import KaggleCLI
from kagglemate.tools.llm_client import simple_prompt


NOTEBOOK_ANALYSIS_PROMPT = """Analyze these Kaggle competition notebooks. For each one, extract:
- model used
- feature engineering techniques
- CV strategy
- key innovations
- whether it's worth studying

Notebook list:
{notebook_json}

Output as JSON array:
[{{"title": "...", "model": "...", "techniques": ["..."], "cv": "...", "verdict": "..."}}]"""


class KaggleProvider(ResearchProvider):
    name = "kaggle"
    label_zh = "Kaggle Notebooks"
    label_en = "Kaggle Notebooks"

    def __init__(self):
        self._last_competition_slug: str = ""

    def search(self, query: str, max_results: int = 20) -> list[SourceItem]:
        """Search Kaggle kernels for a competition.

        Args:
            query: Competition slug (e.g. 'titanic').
            max_results: Number of kernels to fetch.
        """
        slug = query  # For Kaggle, the "query" is the competition slug
        self._last_competition_slug = slug

        kernels = KaggleCLI.list_kernels(slug, sort_by="votes", limit=max_results)

        if not kernels:
            # `kaggle kernels list` may return empty for old or certain competitions.
            # This is a limitation of the Kaggle CLI, not the provider.
            print(f"  [kaggle] ⚠️ No kernels found for '{slug}'. "
                  f"This is expected for legacy or new competitions.")
            return []

        # Batch-analyze with DeepSeek Flash (cheaper)
        notebook_json = json.dumps([
            {
                "title": k.get("title", ""),
                "author": k.get("author", ""),
                "votes": int(k.get("totalVotes", 0) or 0),
                "ref": k.get("ref", ""),
            }
            for k in kernels
        ], indent=2, ensure_ascii=False)

        items: list[SourceItem] = []

        try:
            raw = simple_prompt(
                NOTEBOOK_ANALYSIS_PROMPT.format(notebook_json=notebook_json),
                use_flash=True,
            )
            if "```json" in raw:
                raw = raw.split("```json")[1].split("```")[0]
            elif "```" in raw:
                raw = raw.split("```")[1].split("```")[0]
            analyses = json.loads(raw.strip())

            for i, analysis in enumerate(analyses):
                k = kernels[i] if i < len(kernels) else {}
                items.append(SourceItem(
                    title=k.get("title", analysis.get("title", "Unknown")),
                    source="kaggle",
                    url=f"https://www.kaggle.com/code/{k.get('ref', '')}",
                    snippet=f"Model: {analysis.get('model', '?')}. "
                            f"Techniques: {', '.join(analysis.get('techniques', []))}. "
                            f"CV: {analysis.get('cv', '?')}. {analysis.get('verdict', '')}",
                    authors=k.get("author", ""),
                    relevance_score=int(k.get("totalVotes", 0) or 0) / 100.0,
                    metadata={"ref": k.get("ref", ""), "votes": k.get("totalVotes", 0)},
                ))
        except Exception:
            # Fallback: raw metadata without LLM analysis
            for k in kernels:
                items.append(SourceItem(
                    title=k.get("title", "Unknown"),
                    source="kaggle",
                    url=f"https://www.kaggle.com/code/{k.get('ref', '')}",
                    snippet=f"Author: {k.get('author', '?')}. Votes: {k.get('totalVotes', 0)}",
                    authors=k.get("author", ""),
                    relevance_score=int(k.get("totalVotes", 0) or 0) / 100.0,
                    metadata={"ref": k.get("ref", ""), "votes": k.get("totalVotes", 0)},
                ))

        return items
