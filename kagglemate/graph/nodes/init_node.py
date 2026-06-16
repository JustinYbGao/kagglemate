"""Init Node — Phase 1 entry point.

Creates the competition directory structure, downloads data, and populates
the initial state (files list, paths, competition name).
"""

from __future__ import annotations

import json
import zipfile
from pathlib import Path
from datetime import datetime, timezone

from kagglemate.graph.state import KaggleAgentState
from kagglemate.config import config
from kagglemate.tools.kaggle_cli import KaggleCLI


def run(state: KaggleAgentState) -> dict:
    """Initialize a competition workspace.

    Side effects:
        - Creates competitions/<slug>/... directory tree
        - Downloads and extracts competition data
        - Lists available files

    Returns state updates.
    """
    slug = state["competition_slug"]
    comp_dir = config.COMPETITIONS_DIR / slug

    # ── Create directory tree ──
    dirs = {
        "data": comp_dir / "data" / "raw",
        "notebooks": comp_dir / "notebooks",
        "scripts": comp_dir / "scripts",
        "submissions": comp_dir / "submissions",
        "reports": comp_dir / "reports",
    }
    data_dir = dirs["data"]
    report_dir = dirs["reports"]

    for d in dirs.values():
        d.mkdir(parents=True, exist_ok=True)

    # ── Download data ──
    files: list[dict] = []
    try:
        downloaded = KaggleCLI.download(slug, data_dir)
        files = [
            {"name": f.name, "size_mb": round(f.stat().st_size / 1024 / 1024, 2)}
            for f in downloaded.iterdir()
            if f.is_file() and not f.name.endswith(".zip")
        ]
    except RuntimeError as e:
        return {
            "current_phase": "init",
            "errors": [f"Download failed: {str(e)[:500]}"],
            "data_dir": str(data_dir),
            "report_dir": str(report_dir),
            "files": [],
        }

    # ── Get competition name from metadata or CLI ──
    comp_name = slug  # fallback
    try:
        comps = KaggleCLI.list_competitions(search=slug)
        for c in comps:
            if c.get("ref") == slug:
                comp_name = c.get("title", slug)
                break
    except Exception:
        pass

    return {
        "competition_name": comp_name,
        "data_dir": str(data_dir),
        "report_dir": str(report_dir),
        "submission_dir": str(dirs["submissions"]),
        "script_dir": str(dirs["scripts"]),
        "files": files,
        "current_phase": "init",
    }
