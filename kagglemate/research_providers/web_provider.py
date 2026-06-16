"""Web Search Provider — searches blogs, discussions, and external resources.

Uses DuckDuckGo free search (no API key required).
Falls back gracefully if DDG is unavailable.

Searches are targeted: adds 'kaggle competition' suffix to improve relevance.
"""

from __future__ import annotations

from kagglemate.research_providers.base import ResearchProvider, SourceItem


class WebProvider(ResearchProvider):
    name = "web"
    label_zh = "网络搜索"
    label_en = "Web Search"

    def search(self, query: str, max_results: int = 10) -> list[SourceItem]:
        """Search the web for competition-related content.

        Args:
            query: Search terms. Automatically gets 'kaggle competition' suffix.
            max_results: Number of results to return.
        """
        # Add Kaggle context for better results
        full_query = f"{query} kaggle competition"

        try:
            from ddgs import DDGS
            with DDGS() as ddgs:
                results = list(ddgs.text(full_query, max_results=max_results))
        except ImportError:
            try:
                from duckduckgo_search import DDGS
                with DDGS() as ddgs:
                    results = list(ddgs.text(full_query, max_results=max_results))
            except Exception:
                return []
        except Exception:
            return []

        items: list[SourceItem] = []
        for i, r in enumerate(results):
            items.append(SourceItem(
                title=r.get("title", "Untitled"),
                source="web",
                url=r.get("href", ""),
                snippet=r.get("body", "")[:400],
                relevance_score=None if i >= max_results else 1.0 - (i / max_results),
            ))

        return items
