"""Tests that offline paths do not require optional LLM/LangChain dependencies.

These tests use subprocesses so they can verify import behavior in an
environment where LangChain / OpenAI packages are unavailable, without
uninstalling anything from the current interpreter.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import pytest

PROJECT_ROOT = Path(__file__).parent.parent.parent


# Modules that the offline path must be able to import without LangChain.
_OFFLINE_IMPORTS = [
    "kagglemate",
    "kagglemate.tutor",
    "kagglemate.cv_strategy",
    "kagglemate.strategy_validator",
    "kagglemate.baseline_generator",
    "kagglemate.tools.data_profiler",
    "kagglemate.tools.submission_validator",
]


def _run_in_isolated_import_env(code: str, cwd: Path = PROJECT_ROOT) -> subprocess.CompletedProcess:
    """Run Python code with LangChain/OpenAI packages hidden from import."""
    # Build a small import-hook shim that makes langchain/openai imports fail.
    shim_dir = PROJECT_ROOT / ".pytest_km_import_shim"
    shim_dir.mkdir(exist_ok=True)
    for pkg in ("langchain_core", "langchain_openai", "langchain", "langgraph", "openai"):
        (shim_dir / pkg).mkdir(exist_ok=True)
        (shim_dir / pkg / "__init__.py").write_text(
            f'raise ImportError("{pkg} is not installed in this isolated test env")\n',
            encoding="utf-8",
        )

    env = {
        "PYTHONPATH": str(shim_dir),
        "PYTHONDONTWRITEBYTECODE": "1",
    }
    full_code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
{code}
"""
    return subprocess.run(
        [sys.executable, "-c", full_code],
        cwd=cwd,
        env=env,
        capture_output=True,
        text=True,
    )


def test_offline_modules_import_without_langchain():
    """Offline modules import successfully when LangChain/OpenAI are unavailable."""
    imports = "\n".join(f"import {m}" for m in _OFFLINE_IMPORTS)
    result = _run_in_isolated_import_env(imports)
    if result.returncode != 0:
        pytest.fail(
            f"Offline imports failed without LangChain/OpenAI:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )


def test_tutor_use_llm_false_without_langchain(tmp_path: Path):
    """answer_tutoring_question(use_llm=False) works without LangChain/OpenAI."""
    docs = tmp_path / "docs" / "ml_concepts"
    docs.mkdir(parents=True)
    (docs / "cv.md").write_text("# CV\n\nUse StratifiedKFold for imbalanced targets.", encoding="utf-8")

    code = f"""
from pathlib import Path
from kagglemate.tutor.grounded_tutor import answer_tutoring_question

result = answer_tutoring_question(
    question="Which CV strategy should I use?",
    project_root=Path({str(tmp_path)!r}),
    mode="concept_tutor",
    use_llm=False,
)
assert isinstance(result, dict)
assert "answer" in result
assert result["mode"] == "concept_tutor"
print("OK")
"""
    result = _run_in_isolated_import_env(code)
    if result.returncode != 0:
        pytest.fail(
            f"Grounded tutor offline path failed:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "OK" in result.stdout


def test_benchmark_dry_run_without_langchain():
    """Synthetic benchmark dry-run does not import LangChain/OpenAI."""
    code = """
import sys
# Confirm none of the optional LLM packages leaked into sys.modules.
forbidden = {"langchain_core", "langchain_openai", "langchain", "langgraph", "openai"}
found = [m for m in sys.modules if any(m == pkg or m.startswith(pkg + ".") for pkg in forbidden)]
if found:
    raise AssertionError(f"LLM packages leaked into sys.modules: {found}")
print("OK")
"""
    # We run the benchmark dry-run in a subprocess, then check modules.
    benchmark_code = f"""
import sys
sys.path.insert(0, {str(PROJECT_ROOT)!r})
from benchmarks.run_benchmark import main
import sys
sys.argv = ["run_benchmark.py", "--competition", "titanic", "--synthetic", "--dry-run"]
try:
    rc = main()
except SystemExit as e:
    rc = e.code
{code}
sys.exit(rc)
"""
    result = _run_in_isolated_import_env(benchmark_code)
    if result.returncode != 0:
        pytest.fail(
            f"Benchmark dry-run failed without LangChain/OpenAI:\n"
            f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
        )
    assert "OK" in result.stdout


def test_use_llm_true_missing_dependency_suggests_llm_extra():
    """use_llm=True raises a clear error pointing to the [llm] extra when deps are missing."""
    # Use monkeypatching rather than a subprocess: hide openai/langchain from llm_client.
    import kagglemate.tools.llm_client as llm_client_module

    original_require_chat_openai = llm_client_module._require_chat_openai
    original_require_openai = llm_client_module._require_openai

    def _fake_require_chat_openai():
        raise ImportError("No module named 'langchain_openai'")

    def _fake_require_openai():
        raise ImportError("No module named 'openai'")

    llm_client_module._require_chat_openai = _fake_require_chat_openai
    llm_client_module._require_openai = _fake_require_openai

    try:
        from kagglemate.tutor.grounded_tutor import answer_tutoring_question

        with pytest.raises(RuntimeError) as exc_info:
            answer_tutoring_question(
                question="test",
                project_root=Path.cwd(),
                mode="concept_tutor",
                use_llm=True,
            )
        message = str(exc_info.value)
        assert "[llm]" in message, f"Error message should mention [llm] extra: {message}"
    finally:
        llm_client_module._require_chat_openai = original_require_chat_openai
        llm_client_module._require_openai = original_require_openai
