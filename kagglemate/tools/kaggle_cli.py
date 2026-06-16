"""Kaggle CLI wrapper.

Thin layer around `kaggle` shell commands. Every function returns
structured data (dicts / lists) rather than raw text, so the agent
can consume results without parsing.
"""

from __future__ import annotations

import csv
import io
import json
import subprocess
import zipfile
from pathlib import Path
from typing import Optional

from langchain_core.tools import tool
from pydantic import BaseModel, Field

from kagglemate.config import config


# ── Pydantic models for tool arguments ──


class InitCompetitionInput(BaseModel):
    competition_slug: str = Field(
        description="Kaggle competition slug, e.g. 'titanic', 'playground-series-s5e6'"
    )


class DownloadDataInput(BaseModel):
    competition_slug: str = Field(
        description="Competition slug to download data for"
    )
    target_dir: str = Field(
        default="",
        description="Directory to save data. Defaults to competitions/<slug>/data/raw/",
    )


# ── Core CLI wrapper ──


class KaggleCLI:
    """Run `kaggle` commands and return structured results."""

    @staticmethod
    def _run(cmd: list[str], timeout: int = 60) -> subprocess.CompletedProcess:
        """Run a kaggle CLI command, raise on failure."""
        return subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
        )

    @staticmethod
    def list_competitions(search: str = "", sort_by: str = "recentlyCreated",
                          page_size: int = 50, category: str = "all") -> list[dict]:
        """List competitions with smart defaults for active/recent competitions.

        Args:
            search: Optional search term filter.
            sort_by: Sort order. Default 'recentlyCreated' shows newest first.
                     Options: grouped, prize, earliestDeadline, latestDeadline,
                             numberOfTeams, recentlyCreated
            page_size: Results per page (max 200). Default 50.
            category: Competition category. Default 'all'.
                      Options: all, featured, research, recruitment,
                              gettingStarted, masters, playground

        Returns a list of dicts.
        """
        cmd = [
            "kaggle", "competitions", "list", "--csv",
            "--sort-by", sort_by,
            "--page-size", str(min(page_size, 200)),
            "--category", category,
        ]
        if search:
            cmd.extend(["--search", search])

        result = KaggleCLI._run(cmd)
        if result.returncode != 0:
            raise RuntimeError(f"kaggle CLI error: {result.stderr}")

        reader = csv.DictReader(io.StringIO(result.stdout))
        all_comps = list(reader)

        # Filter out competitions with deadlines > 6 months in the past
        from datetime import datetime
        now = datetime.now()
        cutoff = now.strftime("%Y-%m-%d")
        # Simple: keep comps where deadline >= 2025 (crude but effective)
        active = []
        old = []
        for c in all_comps:
            deadline = c.get("deadline", "")
            if deadline >= "2025":
                active.append(c)
            else:
                old.append(c)

        # Show active first, then old (limited)
        return active + old[:5]
        return list(reader)

    @staticmethod
    def list_files(competition_slug: str) -> list[dict]:
        """List data files for a competition.

        Returns a list of dicts with keys: name, size, creationDate.
        """
        cmd = ["kaggle", "competitions", "files", competition_slug, "--csv"]
        result = KaggleCLI._run(cmd, timeout=30)
        if result.returncode != 0:
            raise RuntimeError(f"kaggle CLI error: {result.stderr}")

        reader = csv.DictReader(io.StringIO(result.stdout))
        return list(reader)

    @staticmethod
    def download(competition_slug: str, target_dir: Path) -> Path:
        """Download and extract competition data. Returns the data directory."""
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        result = KaggleCLI._run(
            ["kaggle", "competitions", "download", "-c", competition_slug,
             "-p", str(target_dir)],
            timeout=300,  # 5 min for large downloads
        )
        if result.returncode != 0:
            raise RuntimeError(f"Download failed: {result.stderr}")

        # Extract all zip files
        for zf_path in target_dir.glob("*.zip"):
            with zipfile.ZipFile(zf_path, "r") as zf:
                zf.extractall(target_dir)

        return target_dir

    @staticmethod
    def list_kernels(competition_slug: str, sort_by: str = "votes",
                     limit: int = 20) -> list[dict]:
        """List public kernels/notebooks for a competition.

        Returns a list of dicts with keys: ref, title, author, lastRunTime,
        totalVotes, medal, etc.
        """
        cmd = [
            "kaggle", "kernels", "list",
            "--competition", competition_slug,
            "--sort-by", sort_by,
            "--csv",
        ]
        result = KaggleCLI._run(cmd, timeout=30)
        if result.returncode != 0:
            # Non-zero is common when there are no kernels yet — don't crash
            return []

        reader = csv.DictReader(io.StringIO(result.stdout))
        kernels = list(reader)
        return kernels[:limit]

    @staticmethod
    def pull_kernel(kernel_ref: str, target_dir: Path) -> Path:
        """Pull a notebook WITH metadata (-m flag is critical).

        kernel_ref format: "username/kernel-name"
        """
        target_dir = Path(target_dir)
        target_dir.mkdir(parents=True, exist_ok=True)

        result = KaggleCLI._run(
            ["kaggle", "kernels", "pull", kernel_ref,
             "-p", str(target_dir), "-m"],
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kernel pull failed: {result.stderr}")

        return target_dir

    @staticmethod
    def push_kernel(kernel_dir: Path) -> dict:
        """Push a kernel to Kaggle. Returns parsed output info."""
        result = KaggleCLI._run(
            ["kaggle", "kernels", "push", "-p", str(kernel_dir)],
            timeout=120,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Kernel push failed: {result.stderr}")

        return {"status": "pushed", "stdout": result.stdout}

    @staticmethod
    def kernel_status(kernel_ref: str) -> dict:
        """Get current status of a kernel."""
        result = KaggleCLI._run(
            ["kaggle", "kernels", "status", kernel_ref],
            timeout=30,
        )
        status = result.stdout.strip() if result.returncode == 0 else "error"
        return {"ref": kernel_ref, "status": status, "stderr": result.stderr}

    @staticmethod
    def submit(competition_slug: str, file_path: Path, message: str = "") -> dict:
        """Submit a prediction file to a competition."""
        cmd = [
            "kaggle", "competitions", "submit",
            competition_slug,
            "-f", str(file_path),
            "-m", message or "kagglemate submission",
        ]
        result = KaggleCLI._run(cmd, timeout=60)
        if result.returncode != 0:
            raise RuntimeError(f"Submission failed: {result.stderr}")

        return {"status": "submitted", "stdout": result.stdout}

    @staticmethod
    def submissions(competition_slug: str) -> list[dict]:
        """List recent submissions for a competition."""
        cmd = ["kaggle", "competitions", "submissions", competition_slug, "--csv"]
        result = KaggleCLI._run(cmd, timeout=30)
        if result.returncode != 0:
            return []

        reader = csv.DictReader(io.StringIO(result.stdout))
        return list(reader)


# ── LangChain-compatible @tool wrappers ──


@tool(args_schema=InitCompetitionInput)
def list_competitions(search: str = "") -> str:
    """List Kaggle competitions, optionally filtered by a search term.

    Use this to find active competitions or check if a competition exists.
    Returns JSON with competition metadata (name, deadline, category, etc.).
    """
    comps = KaggleCLI.list_competitions(search=search)
    return json.dumps(comps, indent=2, ensure_ascii=False)


@tool(args_schema=InitCompetitionInput)
def get_competition_files(competition_slug: str) -> str:
    """List all data files available for a Kaggle competition.

    Use BEFORE downloading to see what files are available (train.csv,
    test.csv, sample_submission.csv, etc.) and their sizes.
    """
    files = KaggleCLI.list_files(competition_slug)
    return json.dumps(files, indent=2, ensure_ascii=False)


@tool(args_schema=DownloadDataInput)
def download_competition_data(competition_slug: str, target_dir: str = "") -> str:
    """Download and extract all data files for a Kaggle competition.

    target_dir defaults to competitions/<slug>/data/raw/ under the project root.
    """
    target = Path(target_dir) if target_dir else (
        config.COMPETITIONS_DIR / competition_slug / "data" / "raw"
    )
    KaggleCLI.download(competition_slug, target)
    # List extracted files
    extracted = [f.name for f in target.iterdir() if f.is_file()]
    return json.dumps({
        "status": "downloaded",
        "target_dir": str(target),
        "files": extracted,
    }, indent=2, ensure_ascii=False)
