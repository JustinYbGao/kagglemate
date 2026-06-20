"""Tests for the offline tutor CLI commands (km tutor / km ask)."""

from __future__ import annotations

from pathlib import Path

import pytest
from typer.testing import CliRunner

# Import the app from the CLI shim
from kagglemate.cli import app

runner = CliRunner()


@pytest.fixture
def project_with_concepts(tmp_path: Path) -> Path:
    """Minimal project with a concept doc for offline tutor tests."""
    concepts = tmp_path / "docs" / "ml_concepts"
    concepts.mkdir(parents=True)
    (concepts / "target_encoding.md").write_text(
        "# Target Encoding\n\n"
        "Target encoding replaces a category with the mean target value.\n\n"
        "## Common failure modes\n\n"
        "Target encoding must be out-of-fold, otherwise it causes leakage.\n",
        encoding="utf-8",
    )
    return tmp_path


def test_km_help_works():
    """km --help renders without importing optional LLM deps."""
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0, result.output
    assert "tutor" in result.output
    assert "ask" in result.output


def test_km_tutor_offline_answers(project_with_concepts: Path):
    """km tutor '...' --offline returns a grounded answer without LLM."""
    result = runner.invoke(app, [
        "tutor",
        "Why does target encoding risk leakage?",
        "--mode", "concept_tutor",
        "--offline",
        "--project-root", str(project_with_concepts),
    ])

    assert result.exit_code == 0, result.output
    assert "Target Encoding" in result.output or "target encoding" in result.output.lower()
    assert "leakage" in result.output.lower()


def test_km_ask_alias_offline_answers(project_with_concepts: Path):
    """km ask is an alias for km tutor and works offline."""
    result = runner.invoke(app, [
        "ask",
        "Why does target encoding risk leakage?",
        "--mode", "concept_tutor",
        "--offline",
        "--project-root", str(project_with_concepts),
    ])

    assert result.exit_code == 0, result.output
    assert "leakage" in result.output.lower()


def test_km_tutor_show_sources(project_with_concepts: Path):
    """--show-sources prints artifact source paths."""
    result = runner.invoke(app, [
        "tutor",
        "target encoding leakage",
        "--mode", "concept_tutor",
        "--offline",
        "--show-sources",
        "--project-root", str(project_with_concepts),
    ])

    assert result.exit_code == 0, result.output
    assert "Sources:" in result.output
    assert "target_encoding.md" in result.output


def test_km_tutor_offline_no_langchain_imported(project_with_concepts: Path):
    """Invoking tutor --offline does not import LangChain / OpenAI."""
    import sys

    # Snapshot modules before invocation
    before = set(sys.modules.keys())
    result = runner.invoke(app, [
        "tutor",
        "target encoding",
        "--mode", "concept_tutor",
        "--offline",
        "--project-root", str(project_with_concepts),
    ])
    after = set(sys.modules.keys())

    assert result.exit_code == 0, result.output
    imported = after - before
    forbidden = {m for m in imported if any(m == pkg or m.startswith(pkg + ".")
                                            for pkg in ("langchain", "langgraph", "openai"))}
    assert not forbidden, f"LLM packages leaked into sys.modules: {sorted(forbidden)}"


def test_km_tutor_online_missing_llm_gives_clear_error(project_with_concepts: Path, monkeypatch):
    """--online without LLM dependencies surfaces a clear [llm] install hint."""
    from kagglemate.tools import llm_client

    # Simulate missing LangChain/OpenAI by making lazy importers fail.
    def _fail():
        raise ImportError("No module named 'openai'")

    monkeypatch.setattr(llm_client, "_require_openai", _fail)
    monkeypatch.setattr(llm_client, "_require_chat_openai", _fail)

    result = runner.invoke(app, [
        "tutor",
        "target encoding",
        "--mode", "concept_tutor",
        "--online",
        "--project-root", str(project_with_concepts),
    ])

    assert result.exit_code != 0
    assert "[llm]" in result.output or "[llm]" in str(result.exception)
