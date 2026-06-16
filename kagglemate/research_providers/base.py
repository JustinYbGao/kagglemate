"""Base class for research providers.

To add a new data source:
1. Subclass ResearchProvider
2. Implement search(query, max_results) → list[SourceItem]
3. Register it in __init__.py

The deep research node calls all registered providers in parallel.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    """A single research finding from any source."""
    title: str = ""
    source: str          # "arxiv" | "kaggle" | "web" | ...
    url: str = ""
    snippet: str = ""    # Short excerpt or abstract
    relevance_score: Optional[float] = None  # 0-1, set by SDK or heuristics
    authors: Optional[str] = None
    date: Optional[str] = None
    metadata: dict = Field(default_factory=dict)  # provider-specific extra data

    def to_markdown(self) -> str:
        """Single-item markdown representation."""
        lines = [f"### {self.title}"]
        if self.authors:
            lines.append(f"- **Authors / 作者**: {self.authors}")
        if self.date:
            lines.append(f"- **Date / 日期**: {self.date}")
        if self.url:
            lines.append(f"- **Source / 来源**: [{self.source}]({self.url})")
        if self.snippet:
            lines.append(f"- **Summary / 摘要**: {self.snippet[:500]}")
        return "\n".join(lines) + "\n"


class ResearchProvider(ABC):
    """Abstract base for research data sources."""

    name: str = "base"       # unique identifier
    label_zh: str = "基础"    # Chinese display label
    label_en: str = "Base"

    @abstractmethod
    def search(self, query: str, max_results: int = 10) -> list[SourceItem]:
        """Search this provider and return findings.

        Args:
            query: Search string (may be provider-specific format).
            max_results: Max number of results to return.

        Returns:
            List of SourceItem, empty list if no results or on error.
        """
        ...
