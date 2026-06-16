"""Session Memory — persistent context across agent restarts.

Saved to ~/.kagglemate/sessions/current.json on exit.
Injected into system prompt on startup.
Archives old sessions with timestamps for debugging.

What it remembers:
- Current competition slug + name + type
- Last 10 actions (tool + brief result)
- User preferences (language, confirmation mode, avoid_submit)
- Active phase (init → research → build → run → evaluate → submit)
- Learning history: concepts explained, notebooks studied, experiments compared
- Summary of last session's progress
"""

from __future__ import annotations

import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional


# ═══════════════════════════════════════════════════════════════════════════════
# Session State Data Model
# ═══════════════════════════════════════════════════════════════════════════════

class SessionState:
    """Mutable session state — saved to disk on every update."""

    def __init__(self):
        self.session_id: str = datetime.now(timezone.utc).strftime("%Y%m%d-%H%M%S")
        self.created_at: str = datetime.now(timezone.utc).isoformat()
        self.updated_at: str = self.created_at
        self.restart_count: int = 0

        # ── Current competition / 当前比赛 ──
        self.current_competition: Optional[str] = None
        self.competition_name: str = ""
        self.competition_type: str = ""

        # ── Recent actions / 最近操作 ──
        self.last_actions: list[dict] = []  # [{tool, slug, summary, timestamp}]

        # ── User preferences / 用户偏好 ──
        self.preferences: dict = {
            "language": "",           # "zh" | "en" | ""
            "super_confirm": False,   # /yesall 模式
            "no_submit": False,       # 用户明确禁止提交
        }

        # ── Learning history / 学习历史 ──
        self.notebooks_studied: list[str] = []   # kernel refs
        self.concepts_explained: list[str] = []  # concept names
        self.comparisons_made: list[dict] = []   # [{id_a, id_b, slug}]

        # ── Session summary / 会话摘要 ──
        self.session_summary: str = ""  # LLM-generated summary of last session

    def to_dict(self) -> dict:
        return {
            "session_id": self.session_id,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "restart_count": self.restart_count,
            "current_competition": self.current_competition,
            "competition_name": self.competition_name,
            "competition_type": self.competition_type,
            "last_actions": self.last_actions[-10:],  # keep last 10
            "preferences": self.preferences,
            "notebooks_studied": self.notebooks_studied[-20:],
            "concepts_explained": self.concepts_explained[-30:],
            "comparisons_made": self.comparisons_made[-10:],
            "session_summary": self.session_summary,
        }

    @classmethod
    def from_dict(cls, data: dict) -> "SessionState":
        state = cls()
        state.session_id = data.get("session_id", state.session_id)
        state.created_at = data.get("created_at", state.created_at)
        state.updated_at = data.get("updated_at", state.updated_at)
        state.restart_count = data.get("restart_count", 0) + 1
        state.current_competition = data.get("current_competition")
        state.competition_name = data.get("competition_name", "")
        state.competition_type = data.get("competition_type", "")
        state.last_actions = data.get("last_actions", [])[-10:]
        state.preferences = data.get("preferences", {})
        state.notebooks_studied = data.get("notebooks_studied", [])[-20:]
        state.concepts_explained = data.get("concepts_explained", [])[-30:]
        state.comparisons_made = data.get("comparisons_made", [])[-10:]
        state.session_summary = data.get("session_summary", "")
        return state


# ═══════════════════════════════════════════════════════════════════════════════
# Persistence Layer
# ═══════════════════════════════════════════════════════════════════════════════

def _sessions_dir() -> Path:
    path = Path.home() / ".kagglemate" / "sessions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _current_path() -> Path:
    return _sessions_dir() / "current.json"


def _archive_path(session_id: str) -> Path:
    return _sessions_dir() / f"{session_id}.json"


def load_session() -> SessionState:
    """Load the last session state, or create a new one."""
    path = _current_path()
    if path.exists():
        try:
            data = json.loads(path.read_text())
            state = SessionState.from_dict(data)
            # Archive the old session
            _archive_session(data)
            return state
        except (json.JSONDecodeError, KeyError):
            pass
    return SessionState()


def save_session(state: SessionState):
    """Save current session state to disk."""
    state.updated_at = datetime.now(timezone.utc).isoformat()
    data = state.to_dict()
    _current_path().write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def _archive_session(data: dict):
    """Archive an old session before overwriting current."""
    sid = data.get("session_id", "unknown")
    archive = _archive_path(sid)
    if not archive.exists():
        archive.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")


def list_archived_sessions() -> list[Path]:
    """Return sorted list of archived session files."""
    return sorted(
        [p for p in _sessions_dir().glob("*.json") if p.name != "current.json"],
        reverse=True,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt Integration / 注入到 System Prompt
# ═══════════════════════════════════════════════════════════════════════════════

def build_memory_context(state: SessionState) -> str:
    """Build a context block to inject into the system prompt.

    Tells the agent: what competition we're on, what we did last time,
    what preferences the user has expressed.
    """
    parts = ["## Session Memory / 会话记忆"]

    # ── Current competition ──
    if state.current_competition:
        parts.append(f"\n**当前比赛 / Current competition**: `{state.current_competition}`")
        if state.competition_name and state.competition_name != state.current_competition:
            parts.append(f"  名称: {state.competition_name}")
        if state.competition_type:
            parts.append(f"  类型: {state.competition_type}")
        parts.append("  用户正在打这个比赛。优先在这个比赛上继续工作。")

    # ── Recent actions ──
    if state.last_actions:
        parts.append(f"\n**最近的操作 / Recent actions** ({len(state.last_actions)} 次):")
        for action in state.last_actions[-8:]:
            tool = action.get("tool", "?")
            slug = action.get("slug", "")
            summary = action.get("summary", "")[:100]
            ts = action.get("timestamp", "")[:10]
            parts.append(f"  - [{ts}] `{tool}` ({slug}): {summary}")

    # ── Preferences ──
    prefs = state.preferences
    pref_lines = []
    if prefs.get("no_submit"):
        pref_lines.append("    ⛔ 用户禁止提交。绝不调用 submit_to_kaggle。")
    if prefs.get("language"):
        pref_lines.append(f"    使用语言: {prefs['language']}")
    if pref_lines:
        parts.append(f"\n**用户偏好 / Preferences**:")
        parts.extend(pref_lines)

    # ── Learning history ──
    if state.notebooks_studied:
        parts.append(f"\n**学习过的 Notebooks**: {', '.join(state.notebooks_studied[-5:])}")
    if state.concepts_explained:
        parts.append(f"\n**讲解过的概念**: {', '.join(state.concepts_explained[-5:])}")

    # ── Summary ──
    if state.session_summary:
        parts.append(f"\n**上次会话摘要 / Last session**: {state.session_summary}")

    # ── Restart hint ──
    if state.restart_count > 0:
        parts.append(f"\n(这是第 {state.restart_count + 1} 次会话。继续之前的工作。)")

    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Action Recording / 操作记录
# ═══════════════════════════════════════════════════════════════════════════════

def record_action(state: SessionState, tool_name: str, slug: str,
                  summary: str = "", success: bool = True):
    """Record a tool invocation to session memory."""
    state.last_actions.append({
        "tool": tool_name,
        "slug": slug,
        "summary": summary[:150],
        "success": success,
        "timestamp": datetime.now(timezone.utc).isoformat()[:19],
    })

    # Update current competition tracking
    if slug and tool_name in ("research_competition", "deep_research",
                               "inspect_competition", "generate_baseline",
                               "run_experiment", "tune_model"):
        state.current_competition = slug

    # Track learning actions
    if tool_name == "explain_notebook":
        state.notebooks_studied.append(slug)
    elif tool_name == "explain_concept":
        state.concepts_explained.append(slug)

    if tool_name == "submit_to_kaggle" and not success:
        # User declined → set preference
        state.preferences["no_submit"] = True

    # Trim
    state.last_actions = state.last_actions[-20:]
    state.notebooks_studied = state.notebooks_studied[-30:]
    state.concepts_explained = state.concepts_explained[-50:]
