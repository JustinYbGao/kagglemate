"""Pure data types shared across KaggleMate.

These TypedDicts have no dependency on LangChain / LangGraph so that offline
modules (data profiling, CV planning, strategy validation, baseline generation)
can import them without pulling in optional LLM libraries.
"""

from __future__ import annotations

from typing import Optional, TypedDict


class FileInfo(TypedDict):
    name: str
    size_mb: float


class ColumnInfo(TypedDict):
    name: str
    dtype: str
    n_missing: int
    missing_pct: float
    n_unique: int


class DataProfile(TypedDict, total=False):
    train_rows: int
    test_rows: int
    columns: list[str]
    target_col: str
    id_col: str
    numerical_cols: list[str]
    categorical_cols: list[str]
    datetime_cols: list[str]
    missing_values: dict[str, float]  # col_name → missing_pct
    target_distribution: Optional[str]
    submission_cols: list[str]
    submission_rows: int
    column_details: list[ColumnInfo]


class NotebookSummary(TypedDict):
    ref: str                # "username/kernel-name"
    title: str
    author: str
    votes: int
    model: str              # e.g. "LightGBM", "XGBoost Ensemble"
    cv_method: str          # e.g. "StratifiedKFold(n=5)"
    lb_score: Optional[float]
    key_techniques: list[str]
    worth_reproducing: bool
    notes: str


class ExperimentRecord(TypedDict, total=False):
    id: int
    name: str
    model: str
    cv_score: float
    cv_std: float
    lb_score: Optional[float]
    metric: str
    params: dict
    features: list[str]
    submission_path: str
    script_path: str
    status: str             # "completed" | "failed" | "running"
    error_message: str
    created_at: str
