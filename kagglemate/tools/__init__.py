"""Tools: wrappers around Kaggle CLI, data profiling, submission validation, etc.

The KaggleCLI class is available immediately.  The LangChain ``@tool`` wrappers
are built lazily via ``get_langchain_tools()`` so that importing this package
does not require LangChain unless an LLM feature is actually used.
"""

from __future__ import annotations

from kagglemate.tools.kaggle_cli import KaggleCLI

__all__ = [
    "KaggleCLI",
    "list_competitions",
    "get_competition_files",
    "download_competition_data",
]


def __getattr__(name: str):
    """Lazy-load LangChain tool wrappers on first access."""
    if name in {"list_competitions", "get_competition_files", "download_competition_data"}:
        from kagglemate.tools.kaggle_cli import get_langchain_tools
        tools = get_langchain_tools()
        return tools[name]
    raise AttributeError(f"module 'kagglemate.tools' has no attribute '{name}'")
