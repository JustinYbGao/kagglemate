"""Kernel Node — Kaggle notebook pull / push / monitor.

Encodes all agentic-kaggle skill kernel workflow patterns:
- Pull with `-m` flag (preserves kernel-metadata.json)
- Metadata validation (id, is_private, competition_sources, internet)
- Push with `-k -v` to prevent orphan submissions
- Monitor with polling (PENDING→RUNNING→COMPLETE/ERROR)
- Structured result parsing from kernel output logs
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path
from typing import Literal

from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.kaggle_cli import KaggleCLI
from kagglemate.config import config


# ── Constants ──

MONITOR_POLL_INTERVAL = 15  # seconds between status checks
MONITOR_MAX_WAIT = 7200      # max wait time (2 hours)


def run(state: KaggleAgentState) -> dict:
    """Execute a kernel action (pull/push/monitor).

    Reads `kernel_action` and `kernel_ref` from state.
    Returns results and any errors.
    """
    action = state.get("kernel_action", "pull")
    kernel_ref = state.get("kernel_ref", "")

    if action == "pull":
        return _pull_kernel(state)
    elif action == "push":
        return _push_kernel(state)
    elif action == "monitor":
        return _monitor_kernel(state)
    elif action == "status":
        return _check_status(state)
    else:
        return {"errors": [f"Unknown kernel action: {action}"], "current_phase": "kernel"}


# ── Pull ──


def _pull_kernel(state: KaggleAgentState) -> dict:
    """Pull a public notebook with metadata preservation.

    Uses `-m` flag (CRITICAL from agentic-kaggle skill) to get
    kernel-metadata.json with all dependencies.
    """
    kernel_ref = state.get("kernel_ref", "")
    slug = state["competition_slug"]

    if not kernel_ref:
        return {"errors": ["No kernel_ref provided for pull"], "current_phase": "kernel"}

    comp_dir = config.COMPETITIONS_DIR / slug
    notebook_dir = comp_dir / "notebooks" / kernel_ref.replace("/", "_")
    notebook_dir.mkdir(parents=True, exist_ok=True)

    _log(f"Pulling: {kernel_ref} → {notebook_dir}")

    try:
        KaggleCLI.pull_kernel(kernel_ref, notebook_dir)
    except RuntimeError as e:
        return {"errors": [f"Kernel pull failed: {e}"], "current_phase": "kernel"}

    # Verify kernel-metadata.json was saved
    metadata_path = notebook_dir / "kernel-metadata.json"
    if not metadata_path.exists():
        return {
            "errors": [
                "kernel-metadata.json not found after pull. "
                "The -m flag may not have worked. Try again."
            ],
            "current_phase": "kernel",
        }

    # Read and validate metadata
    try:
        metadata = json.loads(metadata_path.read_text())
    except json.JSONDecodeError as e:
        return {"errors": [f"Invalid kernel-metadata.json: {e}"], "current_phase": "kernel"}

    # ── Run metadata validation ──
    validation = _validate_metadata(metadata, state)
    _log(f"Metadata validation: {len(validation['warnings'])} warnings, "
         f"{len(validation['errors'])} errors")

    _log(f"Pull complete. Files in {notebook_dir}:")
    for f in sorted(notebook_dir.iterdir()):
        _log(f"  {f.name}")

    return {
        "current_phase": "kernel",
        "errors": validation["errors"],
        "kernel_metadata": metadata,
        "kernel_dir": str(notebook_dir),
    }


# ── Push ──


def _push_kernel(state: KaggleAgentState) -> dict:
    """Push a kernel to Kaggle after metadata validation.

    Uses `-k -v` flags to prevent orphan submissions (from agentic-kaggle skill).
    """
    kernel_dir = state.get("kernel_dir", "")
    if not kernel_dir:
        return {"errors": ["No kernel_dir specified for push"], "current_phase": "kernel"}

    kernel_path = Path(kernel_dir)
    if not kernel_path.exists():
        return {"errors": [f"Kernel directory not found: {kernel_dir}"], "current_phase": "kernel"}

    # ── Pre-push metadata validation ──
    metadata_path = kernel_path / "kernel-metadata.json"
    if not metadata_path.exists():
        return {
            "errors": [
                "No kernel-metadata.json found. Create one before pushing.\n"
                'Minimal example: {"id": "your-username/kernel-name", '
                '"title": "My Kernel", "language": "python", '
                '"kernel_type": "notebook", "is_private": true, '
                '"competition_sources": ["competition-slug"]}'
            ],
            "current_phase": "kernel",
        }

    metadata = json.loads(metadata_path.read_text())
    validation = _validate_metadata(metadata, state)
    if validation["errors"]:
        _log("Metadata validation FAILED:")
        for err in validation["errors"]:
            _log(f"  ✗ {err}")
        return {"errors": validation["errors"], "current_phase": "kernel"}

    _log(f"Pushing kernel from: {kernel_dir}")

    try:
        result = KaggleCLI.push_kernel(kernel_path)
        _log(f"Push: {result.get('stdout', 'OK')[:200]}")
    except RuntimeError as e:
        return {"errors": [f"Kernel push failed: {e}"], "current_phase": "kernel"}

    # Extract the kernel ref from metadata (for monitoring)
    kernel_ref = metadata.get("id", "")

    return {
        "current_phase": "kernel",
        "kernel_ref": kernel_ref,
    }


# ── Monitor ──


def _monitor_kernel(state: KaggleAgentState) -> dict:
    """Poll kernel status until COMPLETE or ERROR.

    On COMPLETE: attempts to parse structured results from output.
    On ERROR: captures error log for diagnosis.
    """
    kernel_ref = state.get("kernel_ref", "")
    timeout = state.get("monitor_timeout", MONITOR_MAX_WAIT)

    if not kernel_ref:
        return {"errors": ["No kernel_ref to monitor"], "current_phase": "kernel"}

    _log(f"Monitoring: {kernel_ref} (polling every {MONITOR_POLL_INTERVAL}s, "
         f"max {timeout // 60}min)")

    elapsed = 0
    last_status = ""
    while elapsed < timeout:
        try:
            result = KaggleCLI.kernel_status(kernel_ref)
        except Exception as e:
            _log(f"Status check failed: {e}. Retrying...")
            time.sleep(MONITOR_POLL_INTERVAL)
            elapsed += MONITOR_POLL_INTERVAL
            continue

        status = result.get("status", "unknown")
        # Normalize — Kaggle CLI may return slightly different strings
        status_lower = status.lower().replace(" ", "")

        if status_lower != last_status:
            _log(f"Status: {status} (elapsed: {elapsed}s)")
            last_status = status_lower

        if "complete" in status_lower:
            _log("✅ Kernel completed!")
            # Try to pull output
            return _handle_kernel_complete(kernel_ref, state)

        elif "error" in status_lower or "failed" in status_lower:
            stderr = result.get("stderr", "")
            _log(f"❌ Kernel failed: {stderr[:300]}")
            return _handle_kernel_error(kernel_ref, stderr, state)

        elif "cancel" in status_lower or "timeout" in status_lower:
            return {
                "errors": [f"Kernel {status_lower}"],
                "current_phase": "kernel",
            }

        time.sleep(MONITOR_POLL_INTERVAL)
        elapsed += MONITOR_POLL_INTERVAL

    return {
        "errors": [f"Kernel monitor timed out after {timeout}s"],
        "current_phase": "kernel",
    }


def _handle_kernel_complete(kernel_ref: str, state: KaggleAgentState) -> dict:
    """Pull the completed kernel output and parse structured results."""
    slug = state["competition_slug"]
    comp_dir = config.COMPETITIONS_DIR / slug
    output_dir = comp_dir / "notebooks" / f"{kernel_ref.replace('/', '_')}_output"

    # Try to pull output
    try:
        KaggleCLI.pull_kernel(kernel_ref, output_dir)
    except RuntimeError:
        _log("Could not pull kernel output (may already be local)")

    # Look for output log
    logs_found = []
    for pattern in ["*.log", "*.txt", "output*"]:
        for f in output_dir.rglob(pattern):
            logs_found.append(f)

    results = {}
    for log_file in logs_found[:5]:
        try:
            content = log_file.read_text()
            parsed = _parse_structured_results(content)
            if parsed:
                results.update(parsed)
                _log(f"Parsed results from {log_file.name}: cv={parsed.get('cv_score')}")
        except Exception:
            pass

    return {
        "current_phase": "kernel",
        "kernel_results": results,
    }


def _handle_kernel_error(kernel_ref: str, stderr: str, state: KaggleAgentState) -> dict:
    """Analyze kernel error and suggest fixes."""
    # Common Kaggle error patterns
    patterns = {
        "module not found": "Missing Python package. Add to dataset_sources or pip install in notebook.",
        "no space left": "Kaggle disk quota exceeded. Reduce output size.",
        "gpu quota": "GPU quota exceeded. Try again later or use CPU.",
        "timeout": "Kernel timed out. Reduce computation or split across multiple kernels.",
        "permission denied": "File permission error. Check path references.",
        "rate limit": "API rate limited. Add retry logic or wait.",
    }

    suggestions = []
    stderr_lower = stderr.lower()
    for pattern, suggestion in patterns.items():
        if pattern in stderr_lower:
            suggestions.append(suggestion)

    return {
        "errors": [f"Kernel failed: {stderr[:500]}"],
        "current_phase": "kernel",
        "error_suggestions": suggestions,
    }


def _check_status(state: KaggleAgentState) -> dict:
    """Quick one-shot status check."""
    kernel_ref = state.get("kernel_ref", "")
    if not kernel_ref:
        return {"errors": ["No kernel_ref provided"], "current_phase": "kernel"}

    try:
        result = KaggleCLI.kernel_status(kernel_ref)
    except Exception as e:
        return {"errors": [f"Status check failed: {e}"], "current_phase": "kernel"}

    return {
        "current_phase": "kernel",
        "kernel_status": result.get("status", "unknown"),
    }


# ── Metadata Validation ──


def _validate_metadata(metadata: dict, state: KaggleAgentState) -> dict:
    """Validate kernel-metadata.json against best practices.

    Returns {"errors": [...], "warnings": [...]}

    Checks (from agentic-kaggle skill):
    - id should be <username>/<kernel-name> (not original author's)
    - is_private should be true
    - competition_sources should include the current competition
    - enable_internet should respect competition rules
    - kernel_type should be "notebook"
    """
    errors: list[str] = []
    warnings: list[str] = []

    username = config.KAGGLE_USERNAME

    # 1. id check
    kernel_id = metadata.get("id", "")
    if not kernel_id:
        errors.append("id is missing — set to '<your-username>/<kernel-name>'")
    elif username and not kernel_id.startswith(f"{username}/"):
        warnings.append(
            f"id is '{kernel_id}' — change to '{username}/<your-kernel-name>' before pushing"
        )

    # 2. is_private
    if not metadata.get("is_private", False):
        errors.append("is_private should be true (prevents accidental public exposure)")

    # 3. competition_sources
    comp_sources = metadata.get("competition_sources", [])
    slug = state["competition_slug"]
    if slug not in comp_sources:
        warnings.append(
            f"competition_sources ({comp_sources}) doesn't include '{slug}'. "
            f"Add it before pushing."
        )

    # 4. enable_internet
    if metadata.get("enable_internet", False):
        warnings.append(
            "enable_internet is true. Verify this is allowed by competition rules."
        )

    # 5. kernel_type
    if metadata.get("kernel_type") not in ("notebook", "script"):
        warnings.append("kernel_type should be 'notebook' or 'script'")

    # 6. language
    if metadata.get("language") not in ("python", "r"):
        warnings.append("language not set — should be 'python'")

    return {"errors": errors, "warnings": warnings}


# ── Structured Result Parsing ──


def _parse_structured_results(text: str) -> dict | None:
    """Extract === RESULTS === JSON block from kernel output.

    Same format as baseline scripts: printed JSON after the marker.
    """
    match = re.search(r"=== RESULTS ===\s*\n(\{.*?\})\s*\n", text, re.DOTALL)
    if not match:
        return None
    try:
        return json.loads(match.group(1).strip())
    except json.JSONDecodeError:
        return None


def _log(msg: str):
    print(f"  [kernel] {msg}")
