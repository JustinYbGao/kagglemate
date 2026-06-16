"""Tools: wrappers around Kaggle CLI, data profiling, submission validation, etc.

Each tool can be called both as a regular Python function and as a
LangChain @tool that the agent can use via function calling.
"""

from kagglemate.tools.kaggle_cli import (
    KaggleCLI,
    list_competitions,
    get_competition_files,
    download_competition_data,
)

__all__ = [
    "KaggleCLI",
    "list_competitions",
    "get_competition_files",
    "download_competition_data",
]
