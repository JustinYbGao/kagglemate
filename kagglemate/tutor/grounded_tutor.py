"""Public API for the grounded tutoring layer.

Provides ``answer_tutoring_question()``, which retrieves local project
artifacts and produces a structured answer that separates facts,
interpretation, uncertainty, and next experiments.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from kagglemate.tutor.context_builder import TutoringContextBuilder
from kagglemate.tutor.experiment_diagnosis import (
    compare_experiments,
    extract_experiment_facts,
)
from kagglemate.tutor.prompts import (
    DETERMINISTIC_ANSWER_TEMPLATE,
    DEFAULT_UNCERTAINTY,
    MODE_SPECIFIC_NEXT_EXPERIMENT,
    build_grounded_prompt,
)
from kagglemate.tutor.sources import SourceChunk


DEFAULT_TOP_K = 6


def _format_confirmed_facts(chunks: list[SourceChunk]) -> list[str]:
    """Extract confirmed facts from retrieved chunks for deterministic mode."""
    facts: list[str] = []
    for chunk in chunks:
        text = chunk.text.strip().replace("\n", " ")
        if len(text) > 240:
            text = text[:237] + "..."
        facts.append(f"[{chunk.source_type} | {chunk.chunk_id}]: {text}")
    return facts


def _format_interpretation(question: str, mode: str, chunks: list[SourceChunk]) -> list[str]:
    """Produce a deterministic, conservative interpretation."""
    if not chunks:
        return ["- No project artifacts were retrieved, so no interpretation can be made."]

    if mode == "experiment_diagnosis":
        return [
            "- The retrieved experiment artifacts describe what was run, but do not by themselves prove why scores differed.",
            "- Compare configs and fold scores side-by-side before drawing causal conclusions.",
        ]
    if mode == "code_walkthrough":
        return [
            "- The retrieved code chunks show implementation details; runtime behavior should be verified with the actual data.",
        ]
    if mode == "concept_tutor":
        return [
            f"- The concept is discussed in the retrieved notes. Apply it to '{question}' using the current competition's data to confirm relevance.",
        ]
    return [
        "- The retrieved artifacts provide context, but competition-specific performance claims require experimental verification.",
    ]


def _build_deterministic_answer(
    question: str,
    mode: str,
    chunks: list[SourceChunk],
) -> dict[str, Any]:
    """Build a structured answer without calling an LLM.

    This path is fully offline and deterministic, making it ideal for tests
    and environments without API keys.
    """
    if mode == "experiment_diagnosis":
        return _build_experiment_diagnosis_answer(question, chunks)

    confirmed_facts = _format_confirmed_facts(chunks)
    interpretations = _format_interpretation(question, mode, chunks)
    uncertainties = [DEFAULT_UNCERTAINTY] if chunks else [
        "- No supporting artifacts were found. Any performance claim would be unverified."
    ]
    next_experiment = [MODE_SPECIFIC_NEXT_EXPERIMENT.get(
        mode,
        "- Run a minimal experiment with the current CV config and record fold scores for comparison.",
    )]

    if not chunks:
        confirmed_section = "_No supporting artifacts found._"
    else:
        confirmed_section = "\n".join(f"- {f}" for f in confirmed_facts)

    answer = DETERMINISTIC_ANSWER_TEMPLATE.format(
        confirmed=confirmed_section,
        interpretation="\n".join(interpretations),
        uncertainty="\n".join(uncertainties),
        next_experiment="\n".join(next_experiment),
    )

    return {
        "answer": answer.strip(),
        "mode": mode,
        "question": question,
        "sources": [_source_to_dict(c) for c in chunks],
        "confirmed_facts": confirmed_facts,
        "interpretations": interpretations,
        "uncertainties": uncertainties,
        "next_verifiable_experiments": next_experiment,
    }


def _build_experiment_diagnosis_answer(
    question: str,
    chunks: list[SourceChunk],
) -> dict[str, Any]:
    """Build a structured experiment-diagnosis answer grounded in artifacts.

    Separates observed facts from interpretation, explicitly marks missing
    evidence, and never claims causation or leaderboard performance without
    artifact support.
    """
    facts = extract_experiment_facts(chunks)
    comparison = compare_experiments(facts)

    confirmed_facts: list[str] = []
    for fact in facts:
        line = (
            f"Experiment {fact.get('experiment_id')} ({fact.get('experiment_name')}): "
            f"status={fact.get('status')}, cv_score={fact.get('cv_score')}, "
            f"metric={fact.get('metric')}, strategy={fact.get('cv_strategy')}"
        )
        confirmed_facts.append(line)

    interpretations: list[str] = []
    if comparison["best_by_cv"]:
        best = comparison["best_by_cv"]
        interpretations.append(
            f"- Highest CV score observed: {best['cv_score']} "
            f"(experiment {best.get('experiment_id')}). "
            "This does not imply leaderboard performance."
        )
    if len(facts) == 1:
        interpretations.append(
            "- Only one experiment was found; pairwise comparison is not possible."
        )
    elif len(facts) > 1 and comparison.get("score_range"):
        rng = comparison["score_range"]
        interpretations.append(
            f"- Score range across {len(facts)} experiments: "
            f"{rng['min']} to {rng['max']} (delta {rng['delta']})."
        )

    uncertainties: list[str] = []
    if comparison["missing_artifacts"]:
        uncertainties.append(
            "- Missing artifacts limit diagnosis: "
            f"{', '.join(sorted(comparison['missing_artifacts']))}."
        )
    if facts and all(fact.get("lb_score") is None for fact in facts):
        uncertainties.append(
            "- No leaderboard score is recorded in the artifacts; "
            "do not assume CV translates to public/private LB."
        )
    if len(facts) == 1:
        uncertainties.append(
            "- Cannot determine whether observed performance is due to the method "
            "or random variation without a second experiment."
        )
    if not facts:
        uncertainties.append(
            "- No experiment artifacts were retrieved; any causal claim would be unverified."
        )

    next_experiments: list[str] = [
        MODE_SPECIFIC_NEXT_EXPERIMENT.get(
            "experiment_diagnosis",
            "- Run a controlled ablation and compare fold_scores.json files.",
        )
    ]
    if "fold_scores" in comparison["missing_artifacts"]:
        next_experiments.append(
            "- Re-run with fold_scores_path persisted to enable per-fold analysis."
        )
    if "oof" in comparison["missing_artifacts"]:
        next_experiments.append(
            "- Persist OOF predictions to enable leakage and consistency checks."
        )
    if "strategy_validation" in comparison["missing_artifacts"]:
        next_experiments.append(
            "- Generate a strategy_validation_report to confirm the feature pipeline."
        )
    if "submission_validation" in comparison["missing_artifacts"]:
        next_experiments.append(
            "- Generate a submission_validation_report to confirm submission schema."
        )

    if not facts:
        facts_section = "_No experiment artifacts found._"
        diagnosis_section = "_No diagnosis possible without experiment artifacts._"
    else:
        facts_section = "\n".join(f"- {f}" for f in confirmed_facts)
        diagnosis_section = "\n".join(interpretations)

    answer_parts = [
        "## Experiment facts from artifacts",
        facts_section,
        "",
        "## Diagnosis",
        diagnosis_section,
        "",
        "## What cannot be concluded yet",
        "\n".join(uncertainties) if uncertainties else "_None._",
        "",
        "## Next verifiable experiments",
        "\n".join(next_experiments),
    ]
    answer = "\n".join(answer_parts)

    return {
        "answer": answer.strip(),
        "mode": "experiment_diagnosis",
        "question": question,
        "sources": [_source_to_dict(c) for c in chunks],
        "confirmed_facts": confirmed_facts,
        "interpretations": interpretations,
        "uncertainties": uncertainties,
        "next_verifiable_experiments": next_experiments,
    }


def _source_to_dict(chunk: SourceChunk) -> dict[str, Any]:
    """Serialize a SourceChunk for API output."""
    return {
        "source_path": str(chunk.source_path),
        "source_type": chunk.source_type,
        "chunk_id": chunk.chunk_id,
        "line_start": chunk.line_start,
        "line_end": chunk.line_end,
        "score": chunk.metadata.get("score") if chunk.metadata else None,
    }


def answer_tutoring_question(
    question: str,
    project_root: Path,
    competition_slug: str | None = None,
    artifact_dirs: list[Path] | None = None,
    mode: str = "grounded_explanation",
    use_llm: bool = False,
    top_k: int = DEFAULT_TOP_K,
) -> dict[str, Any]:
    """Answer a tutoring question grounded in local project artifacts.

    Args:
        question: The user's question.
        project_root: Root of the KaggleMate project.
        competition_slug: Optional competition slug to narrow artifact search.
        artifact_dirs: Optional extra directories to scan.
        mode: Tutoring mode. One of ``grounded_explanation``,
            ``code_walkthrough``, ``experiment_diagnosis``, ``concept_tutor``.
        use_llm: If True, synthesize with the configured LLM provider.  If
            False (default), return a deterministic structured answer.
        top_k: Number of artifact chunks to retrieve.

    Returns:
        Dictionary with ``answer``, ``mode``, ``question``, ``sources``,
        ``confirmed_facts``, ``interpretations``, ``uncertainties``, and
        ``next_verifiable_experiments``.
    """
    builder = TutoringContextBuilder(
        project_root=Path(project_root),
        competition_slug=competition_slug,
        artifact_dirs=artifact_dirs,
        include_concept_docs=True,
    )
    chunks = builder.retrieve(query=question, mode=mode, top_k=top_k)

    if not use_llm:
        return _build_deterministic_answer(question, mode, chunks)

    prompt = build_grounded_prompt(
        question=question,
        mode=mode,
        chunks=chunks,
        competition_slug=competition_slug,
    )

    try:
        from kagglemate.tools.llm_client import simple_prompt

        raw_answer = simple_prompt(prompt)
    except (ImportError, ModuleNotFoundError, RuntimeError) as exc:
        # Missing optional dependency or LLM client misconfiguration — surface clearly.
        msg = str(exc)
        if "[llm]" not in msg and "Install with:" not in msg:
            msg = (
                "LLM synthesis is unavailable. "
                "Install LLM dependencies with: pip install -e '.[llm]'"
            )
        raise RuntimeError(msg) from exc
    except Exception as exc:
        # Actual LLM call failure (e.g. API error) — fall back deterministically.
        raw_answer = (
            "LLM synthesis is unavailable. Falling back to deterministic summary.\n\n"
            + _build_deterministic_answer(question, mode, chunks)["answer"]
        )

    return {
        "answer": raw_answer.strip(),
        "mode": mode,
        "question": question,
        "sources": [_source_to_dict(c) for c in chunks],
        "confirmed_facts": [],
        "interpretations": [],
        "uncertainties": [],
        "next_verifiable_experiments": [],
    }


def format_answer_as_json(result: dict[str, Any]) -> str:
    """Pretty-print a tutoring result as JSON."""
    return json.dumps(result, indent=2, ensure_ascii=False)
