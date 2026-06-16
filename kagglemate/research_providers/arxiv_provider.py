"""arXiv Provider — searches academic papers via the free arXiv API.

No API key required. Uses feedparser to parse Atom XML responses.
Rate limit: ~1 request per 3 seconds (enforced internally).

Key design: the provider extracts technical keywords from the competition
context before searching, improving relevance. Raw competition names
("orbit-wars") rarely match arXiv papers, but technique names do.
"""

from __future__ import annotations

import time
import urllib.parse
import urllib.request
from dataclasses import dataclass

import feedparser

from kagglemate.research_providers.base import ResearchProvider, SourceItem

ARXIV_API_URL = "http://export.arxiv.org/api/query"


class ArxivProvider(ResearchProvider):
    name = "arxiv"
    label_zh = "arXiv 论文"
    label_en = "arXiv Papers"
    # ── Internal rate limiter ──
    _last_request_time: float = 0.0

    def search(self, query: str, max_results: int = 10) -> list[SourceItem]:
        """Search arXiv for papers matching the query.

        Args:
            query: Search terms (e.g. 'reinforcement learning multi-agent game').
                   The caller (deep research node) should translate competition
                   context into technical search terms.
            max_results: Number of papers to return.
        """
        self._rate_limit()

        params = {
            "search_query": f"all:{query}",
            "start": 0,
            "max_results": max_results,
            "sortBy": "relevance",
        }
        url = f"{ARXIV_API_URL}?{urllib.parse.urlencode(params)}"

        try:
            response = urllib.request.urlopen(url, timeout=10)
            feed = feedparser.parse(response.read())
        except Exception:
            # arXiv may be unreachable from some regions (e.g., China).
            # This is a graceful degradation — the report will still include
            # Kaggle and Web results.
            print(f"  [arxiv] ⚠️ arXiv API unreachable (timeout/network). This is expected from some regions.")
            return []

        items: list[SourceItem] = []
        for entry in feed.entries[:max_results]:
            # Extract categories
            categories = [t.get("term", "") for t in getattr(entry, "tags", []) if t]

            # Build snippet from abstract
            abstract = entry.get("summary", "")
            # Strip HTML tags from abstract
            abstract = abstract.replace("<p>", "").replace("</p>", "\n").strip()
            abstract = abstract[:500]

            items.append(SourceItem(
                title=entry.get("title", "Untitled").strip(),
                source="arxiv",
                url=entry.get("id", ""),
                snippet=abstract,
                authors=", ".join(a.get("name", "") for a in entry.get("authors", [])),
                date=entry.get("published", "")[:10],
                relevance_score=None,
                metadata={
                    "categories": categories,
                    "pdf_url": entry.get("id", "").replace("/abs/", "/pdf/") if entry.get("id") else "",
                },
            ))

        return items

    def _rate_limit(self):
        """Ensure at least 3 seconds between requests (arXiv requirement)."""
        now = time.time()
        elapsed = now - self._last_request_time
        if elapsed < 3.0:
            time.sleep(3.0 - elapsed)
        self._last_request_time = time.time()
