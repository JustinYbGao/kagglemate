"""Experiment store — SQLite-backed experiment tracking.

One database file per competition: competitions/<slug>/experiments.db
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from datetime import datetime, timezone
from typing import Optional


SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS experiments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_name TEXT NOT NULL,
    competition_slug TEXT NOT NULL,
    task_type TEXT,
    target_column TEXT,
    id_column TEXT,
    model_name TEXT NOT NULL DEFAULT 'Unknown',
    cv_score REAL,
    cv_std REAL,
    lb_score REAL,
    metric TEXT DEFAULT 'unknown',
    cv_folds INTEGER DEFAULT 5,
    cv_strategy TEXT,
    features TEXT,
    params TEXT,
    feature_importance TEXT,
    fold_scores TEXT,
    oof_path TEXT,
    fold_scores_path TEXT,
    config_path TEXT,
    strategy_validation_report_path TEXT,
    submission_validation_report_path TEXT,
    benchmark_result_path TEXT,
    runtime_seconds REAL,
    script_hash TEXT,
    submission_hash TEXT,
    submission_path TEXT,
    script_path TEXT,
    report_path TEXT,
    notes TEXT,
    status TEXT DEFAULT 'completed',
    error_message TEXT,
    created_at TEXT DEFAULT (datetime('now')),
    lb_updated_at TEXT
);

CREATE INDEX IF NOT EXISTS idx_exp_competition ON experiments(competition_slug);
CREATE INDEX IF NOT EXISTS idx_exp_created ON experiments(created_at);
CREATE INDEX IF NOT EXISTS idx_exp_cv_score ON experiments(cv_score);
"""

# Columns added after initial schema release — applied idempotently via _migrate().
MIGRATIONS = [
    ("oof_path", "TEXT"),
    ("fold_scores_path", "TEXT"),
    ("config_path", "TEXT"),
    ("runtime_seconds", "REAL"),
    ("script_hash", "TEXT"),
    ("submission_hash", "TEXT"),
    ("task_type", "TEXT"),
    ("target_column", "TEXT"),
    ("id_column", "TEXT"),
    ("cv_strategy", "TEXT"),
    ("strategy_validation_report_path", "TEXT"),
    ("submission_validation_report_path", "TEXT"),
    ("benchmark_result_path", "TEXT"),
]

# Indexes that depend on migrated columns.
MIGRATION_INDEXES = [
    "CREATE INDEX IF NOT EXISTS idx_exp_script_hash ON experiments(script_hash)",
]


class ExperimentStore:
    """CRUD for experiment records in a competition-specific SQLite database."""

    def __init__(self, competition_slug: str, db_path: Optional[str | Path] = None):
        self.slug = competition_slug
        if db_path is None:
            db_path = f"competitions/{competition_slug}/experiments.db"
        self.db_path = Path(db_path)

    def _conn(self) -> sqlite3.Connection:
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        conn.executescript(SCHEMA_SQL)
        conn.commit()
        self._migrate(conn)
        return conn

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Idempotently add columns introduced in newer schema versions."""
        existing = {row["name"] for row in conn.execute("PRAGMA table_info(experiments)")}
        for column, dtype in MIGRATIONS:
            if column not in existing:
                try:
                    conn.execute(f"ALTER TABLE experiments ADD COLUMN {column} {dtype}")
                except sqlite3.OperationalError:
                    pass  # Column may already exist despite PRAGMA race
        for idx_sql in MIGRATION_INDEXES:
            try:
                conn.execute(idx_sql)
            except sqlite3.OperationalError:
                pass
        conn.commit()

    # ── Create ──

    def insert(self, record: dict) -> int:
        """Insert a new experiment record. Returns the new row ID."""
        conn = self._conn()
        row = {
            "experiment_name": record.get("name") or record.get("experiment_name", "unnamed"),
            "competition_slug": self.slug,
            "task_type": record.get("task_type", ""),
            "target_column": record.get("target_column", ""),
            "id_column": record.get("id_column", ""),
            "model_name": record.get("model_name") or record.get("model", "Unknown"),
            "cv_score": record.get("cv_score"),
            "cv_std": record.get("cv_std"),
            "lb_score": record.get("lb_score"),
            "metric": record.get("metric", "unknown"),
            "cv_folds": record.get("cv_folds") or record.get("n_folds", 5),
            "cv_strategy": record.get("cv_strategy", ""),
            "features": self._json(record.get("features", [])),
            "params": self._json(record.get("params", {})),
            "feature_importance": self._json(record.get("feature_importance", [])),
            "fold_scores": self._json(record.get("fold_scores", [])),
            "oof_path": record.get("oof_path", ""),
            "fold_scores_path": record.get("fold_scores_path", ""),
            "config_path": record.get("config_path", ""),
            "strategy_validation_report_path": record.get("strategy_validation_report_path", ""),
            "submission_validation_report_path": record.get("submission_validation_report_path", ""),
            "benchmark_result_path": record.get("benchmark_result_path", ""),
            "runtime_seconds": record.get("runtime_seconds"),
            "script_hash": record.get("script_hash", ""),
            "submission_hash": record.get("submission_hash", ""),
            "submission_path": record.get("submission_path", ""),
            "script_path": record.get("script_path", ""),
            "report_path": record.get("report_path", ""),
            "notes": record.get("notes", ""),
            "status": record.get("status", "completed"),
            "error_message": record.get("error_message", ""),
        }
        columns = ", ".join(row.keys())
        placeholders = ", ".join("?" for _ in row)
        values = list(row.values())

        cur = conn.execute(
            f"INSERT INTO experiments ({columns}) VALUES ({placeholders})",
            values,
        )
        conn.commit()
        exp_id = cur.lastrowid
        conn.close()
        return exp_id

    # ── Read ──

    def get(self, exp_id: int) -> Optional[dict]:
        """Get a single experiment by ID."""
        conn = self._conn()
        row = conn.execute("SELECT * FROM experiments WHERE id = ?", (exp_id,)).fetchone()
        conn.close()
        return self._row_to_dict(row) if row else None

    def get_best(self) -> Optional[dict]:
        """Get the experiment with the highest CV score."""
        conn = self._conn()
        row = conn.execute(
            "SELECT * FROM experiments WHERE competition_slug = ? "
            "AND status = 'completed' AND cv_score IS NOT NULL "
            "ORDER BY cv_score DESC LIMIT 1",
            (self.slug,),
        ).fetchone()
        conn.close()
        return self._row_to_dict(row) if row else None

    def list_all(self, limit: int = 50, offset: int = 0) -> list[dict]:
        """List all experiments for this competition, newest first."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT * FROM experiments WHERE competition_slug = ? "
            "ORDER BY created_at DESC LIMIT ? OFFSET ?",
            (self.slug, limit, offset),
        ).fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def count(self) -> int:
        """Total number of experiments for this competition."""
        conn = self._conn()
        row = conn.execute(
            "SELECT COUNT(*) as n FROM experiments WHERE competition_slug = ?",
            (self.slug,),
        ).fetchone()
        conn.close()
        return row["n"] if row else 0

    # ── Update ──

    def update_lb(self, exp_id: int, lb_score: float) -> bool:
        """Record a leaderboard score for an experiment."""
        conn = self._conn()
        conn.execute(
            "UPDATE experiments SET lb_score = ?, lb_updated_at = datetime('now') WHERE id = ?",
            (lb_score, exp_id),
        )
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return affected > 0

    def update_field(self, exp_id: int, field: str, value) -> bool:
        """Update a single field on an experiment record."""
        conn = self._conn()
        conn.execute(
            f"UPDATE experiments SET {field} = ? WHERE id = ?",
            (value if not isinstance(value, (list, dict)) else self._json(value), exp_id),
        )
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return affected > 0

    def set_status(self, exp_id: int, status: str, error_msg: str = "") -> bool:
        """Mark an experiment as completed/failed/running."""
        conn = self._conn()
        conn.execute(
            "UPDATE experiments SET status = ?, error_message = ? WHERE id = ?",
            (status, error_msg, exp_id),
        )
        conn.commit()
        affected = conn.total_changes
        conn.close()
        return affected > 0

    # ── Compare ──

    def compare(self, ids: list[int]) -> list[dict]:
        """Get multiple experiments side-by-side for comparison."""
        conn = self._conn()
        placeholders = ", ".join("?" for _ in ids)
        rows = conn.execute(
            f"SELECT * FROM experiments WHERE id IN ({placeholders})",
            ids,
        ).fetchall()
        conn.close()
        return [self._row_to_dict(r) for r in rows]

    def cv_lb_gap(self) -> Optional[float]:
        """Best CV score minus its LB score (detects overfitting)."""
        best = self.get_best()
        if best and best.get("lb_score") is not None:
            return best["cv_score"] - best["lb_score"]
        return None

    def list_submission_hashes(self) -> list[str]:
        """Return sha256 hashes of all previous completed submissions."""
        conn = self._conn()
        rows = conn.execute(
            "SELECT submission_hash FROM experiments WHERE competition_slug = ? AND submission_hash IS NOT NULL",
            (self.slug,),
        ).fetchall()
        conn.close()
        return [r["submission_hash"] for r in rows if r["submission_hash"]]

    # ── Helpers ──

    def _row_to_dict(self, row: sqlite3.Row) -> dict:
        d = dict(row)
        # Parse JSON fields
        for field in ("features", "params", "feature_importance", "fold_scores"):
            if d.get(field):
                try:
                    d[field] = json.loads(d[field])
                except (json.JSONDecodeError, TypeError):
                    pass
        return d

    @staticmethod
    def _json(value) -> str:
        return json.dumps(value, ensure_ascii=False)
