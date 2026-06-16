"""Submit Node — validates and optionally submits predictions to Kaggle.

The Human Gate lives in the CLI (main.py). This node handles the
graph-based path: validate → flag for human review → submit on approval.

Key design choice from agentic-kaggle skill:
- Submission is NEVER automatic — always requires human confirmation.
- Before submit: verify format, wait for score stabilization (4+ hours rule).
- After submit: record LB score back to experiments.db.
"""

from __future__ import annotations

from pathlib import Path

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.submission_validator import validate
from kagglemate.tools.kaggle_cli import KaggleCLI
from kagglemate.memory.experiment_store import ExperimentStore


def run(state: KaggleAgentState) -> dict:
    """Run submission validation and optional submission.

    Phase 1 (human_approved=False): validate + generate preview
    Phase 2 (human_approved=True): submit to Kaggle + record LB
    """
    slug = state["competition_slug"]
    sub_path = state.get("submission_file", "")
    exp = state.get("current_experiment") or {}

    if not sub_path:
        # Try to auto-detect from current experiment
        if exp.get("submission_path"):
            sub_path = exp["submission_path"]
        else:
            return {
                "errors": ["No submission file specified. Use --file to specify one."],
                "current_phase": "submit",
            }

    if not Path(sub_path).exists():
        return {
            "errors": [f"Submission file not found: {sub_path}"],
            "current_phase": "submit",
        }

    data_dir = state.get("data_dir", "")

    # ── Step 1: Always validate first ──
    vr = validate(sub_path, data_dir)

    if not vr.is_valid:
        errors = [e for e in vr.errors]
        return {
            "errors": errors,
            "current_phase": "submit",
            "submission_preview": _format_preview(state, sub_path, vr),
        }

    # ── Step 2: Human Gate check ──
    if not state.get("human_approved"):
        # Generate preview and wait for human approval
        preview = _format_preview(state, sub_path, vr)
        _log("Awaiting human approval...")
        _log(preview)

        return {
            "human_approval_required": True,
            "submission_preview": preview,
            "submission_file": sub_path,
            "current_phase": "submit",
        }

    # ── Step 3: Approved — submit! ──
    message = state.get("submission_message", "kagglemate submission")
    _log(f"Submitting: {sub_path} → {slug}")

    try:
        result = KaggleCLI.submit(slug, Path(sub_path), message)
        _log(f"Submitted: {result.get('stdout', 'OK')[:200]}")
    except RuntimeError as e:
        return {
            "errors": [f"Submission failed: {e}"],
            "current_phase": "submit",
        }

    # ── Step 4: Record in experiments.db ──
    exp_id = exp.get("id")
    if exp_id:
        store = ExperimentStore(slug)
        store.update_field(exp_id, "submission_path", sub_path)
        _log(f"Linked submission to experiment #{exp_id}")

    return {
        "human_approved": False,  # reset for next time
        "human_approval_required": False,
        "current_phase": "complete",
    }


def _format_preview(state: KaggleAgentState, sub_path: str, vr) -> str:
    """Format a human-readable submission preview."""
    exp = state.get("current_experiment") or {}
    lines = [
        "=" * 60,
        "         SUBMISSION PREVIEW — PLEASE REVIEW",
        "=" * 60,
        "",
        f"  Competition:   {state['competition_slug']}",
        f"  File:          {sub_path}",
        f"  Experiment:    {exp.get('name', 'N/A')}",
        f"  CV Score:      {exp.get('cv_score', 'N/A')}",
        f"  Model:         {exp.get('model', 'N/A')}",
        "",
        "  Validation:",
    ]
    for check in vr.checks:
        icon = "✓" if check.passed else "✗"
        lines.append(f"    {icon} {check.check}: {check.detail[:80]}")

    if vr.warnings:
        lines.append("")
        lines.append("  Warnings:")
        for w in vr.warnings:
            lines.append(f"    ⚠ {w[:100]}")

    lines += [
        "",
        "  ╔════════════════════════════════════════════════╗",
        "  ║  ⚠  This will use a Kaggle submission slot.  ║",
        "  ║  Have you reviewed the rules checklist?       ║",
        "  ║  Wait 4+ hours for score stabilization.      ║",
        "  ╚════════════════════════════════════════════════╝",
        "",
        "  Type YES to confirm submission:",
    ]
    return "\n".join(lines)


def _log(msg: str):
    print(f"  [submit] {msg}")
