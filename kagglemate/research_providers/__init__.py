"""Research Providers — pluggable data sources for deep research.

Each provider implements: search(query, max_results) → list[SourceItem]
Add a new provider by subclassing ResearchProvider and registering it.
"""

from kagglemate.research_providers.base import ResearchProvider, SourceItem
from kagglemate.research_providers.kaggle_provider import KaggleProvider
from kagglemate.research_providers.arxiv_provider import ArxivProvider
from kagglemate.research_providers.web_provider import WebProvider

__all__ = [
    "ResearchProvider",
    "SourceItem",
    "KaggleProvider",
    "ArxivProvider",
    "WebProvider",
]
