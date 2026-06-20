"""KaggleAgentState — the shared state schema for all LangGraph nodes.

Every node in the graph reads from and writes to this TypedDict.
Annotated fields with reducers (operator.add, add_messages) allow
multiple nodes to append without overwriting each other.
"""

from typing import TypedDict, Annotated, Sequence, Optional, Literal
import operator

from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

from kagglemate.types import (
    ColumnInfo,
    DataProfile,
    ExperimentRecord,
    FileInfo,
    NotebookSummary,
)


# ── Main Graph State ──


class KaggleAgentState(TypedDict, total=False):
    # ── Message history (append-only via add_messages reducer) ──
    messages: Annotated[Sequence[BaseMessage], add_messages]

    # ── Competition identity ──
    competition_slug: str
    competition_name: str
    competition_type: Literal[
        "tabular_classification", "tabular_regression",
        "image", "text", "time_series", "other",
    ]
    evaluation_metric: str
    files: list[FileInfo]

    # ── Paths ──
    data_dir: str
    report_dir: str
    submission_dir: str
    script_dir: str

    # ── Data analysis ──
    data_profile: Optional[DataProfile]
    submission_cols_known: bool

    # ── Research ──
    notebook_summaries: list[NotebookSummary]
    research_complete: bool

    # ── Generated documents ──
    spec_path: Optional[str]
    research_summary_path: Optional[str]
    rules_checklist_path: Optional[str]

    # ── Experiments ──
    current_experiment: Optional[ExperimentRecord]
    all_experiments: Annotated[list[ExperimentRecord], operator.add]
    best_cv_score: float
    best_lb_score: float
    cv_plan: Optional[dict]

    # ── Flow control ──
    current_phase: Literal[
        "init", "analyze", "research", "plan",
        "build", "run", "evaluate", "suggest",
        "kernel", "submit", "complete",
    ]
    errors: Annotated[list[str], operator.add]
    human_approval_required: bool
    human_approved: bool

    # ── Kernel operations (Phase 4) ──
    kernel_action: Optional[str]        # "pull" | "push" | "monitor" | "status"
    kernel_ref: Optional[str]           # "username/kernel-name"
    kernel_dir: Optional[str]           # local path to kernel directory
    kernel_metadata: Optional[dict]     # parsed kernel-metadata.json
    kernel_status: Optional[str]        # "complete" | "error" | "running" | ...
    kernel_results: Optional[dict]      # parsed structured results
    error_suggestions: Optional[list[str]]  # LLM-suggested fixes for kernel errors
    monitor_timeout: Optional[int]      # max seconds for kernel monitoring

    # ── Submission (Phase 5) ──
    submission_file: Optional[str]      # path to submission CSV
    submission_message: Optional[str]   # message for Kaggle submission
    submission_preview: Optional[str]   # preview text for human approval

    # ── Advanced (Phase 6) ──
    tune_trials: Optional[int]          # number of Optuna trials (default 50)
    ensemble_exp_ids: Optional[list[int]]  # experiment IDs to blend
    ensemble_method: Optional[str]      # "simple_average" | "weighted_average" | "rank_average"

    # ── Internal (not part of user-facing state) ──
    _should_continue: Optional[bool]    # conditional edge routing
