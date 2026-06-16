"""Graph builder — assembles the KaggleAgent StateGraph.

Phase 1: Init → Analyze → Research → Plan → END
Phase 2: ... → Plan → Baseline → Run → Evaluate → END
Phase 3: ... → Evaluate → (conditional) → Suggest → END
         Evaluate routes to "suggest" if should_continue, else END.
"""

from __future__ import annotations

from contextlib import contextmanager
from typing import Iterator

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.sqlite import SqliteSaver

from kagglemate.graph.state import KaggleAgentState
from kagglemate.graph.nodes import (
    init_node, analyze_node, research_node, plan_node,
    baseline_node, run_node, evaluate_node, suggest_node,
)
from kagglemate.config import config


# ── Conditional edge logic ──


def _after_evaluate(state: KaggleAgentState) -> str:
    """Route from evaluate: → suggest (continue) or → END (stop)."""
    should_continue = state.get("_should_continue", True)
    return "suggest" if should_continue else END


# ── Phase 1: Research-only Graph ──


def build_research_graph() -> StateGraph:
    """Phase 1: Init → Analyze → Research → Plan → END"""
    builder = StateGraph(KaggleAgentState)

    builder.add_node("init", init_node.run)
    builder.add_node("analyze", analyze_node.run)
    builder.add_node("research", research_node.run)
    builder.add_node("plan", plan_node.run)

    builder.set_entry_point("init")
    builder.add_edge("init", "analyze")
    builder.add_edge("analyze", "research")
    builder.add_edge("research", "plan")
    builder.add_edge("plan", END)

    return builder


# ── Phase 3: Full Graph (with Evaluate → Suggest conditional edge) ──


def build_full_graph() -> StateGraph:
    """Phase 3: Init → Analyze → Research → Plan → Baseline → Run
                         → Evaluate → (conditional) → Suggest → END

    The conditional edge at Evaluate:
    - If should_continue → route to "suggest" node (then END)
    - If too many failures or user stop → route directly to END
    """
    builder = StateGraph(KaggleAgentState)

    # ── All nodes ──
    builder.add_node("init", init_node.run)
    builder.add_node("analyze", analyze_node.run)
    builder.add_node("research", research_node.run)
    builder.add_node("plan", plan_node.run)
    builder.add_node("baseline", baseline_node.run)
    builder.add_node("run", run_node.run)
    builder.add_node("evaluate", evaluate_node.run)
    builder.add_node("suggest", suggest_node.run)

    # ── Edges ──
    builder.set_entry_point("init")
    builder.add_edge("init", "analyze")
    builder.add_edge("analyze", "research")
    builder.add_edge("research", "plan")
    builder.add_edge("plan", "baseline")
    builder.add_edge("baseline", "run")
    builder.add_edge("run", "evaluate")

    # Conditional: evaluate → suggest or END
    builder.add_conditional_edges("evaluate", _after_evaluate, {
        "suggest": "suggest",
        END: END,
    })

    builder.add_edge("suggest", END)

    return builder


def compile_graph(checkpointer=None):
    """Compile with optional checkpointer. Uses MemorySaver if none given."""
    builder = build_full_graph()
    if checkpointer is None:
        checkpointer = MemorySaver()
    return builder.compile(checkpointer=checkpointer)


@contextmanager
def get_full_graph() -> Iterator:
    """Context manager: full graph with SqliteSaver persistence."""
    builder = build_full_graph()
    with SqliteSaver.from_conn_string(config.CHECKPOINT_DB) as checkpointer:
        yield builder.compile(checkpointer=checkpointer)


@contextmanager
def get_research_graph() -> Iterator:
    """Context manager: research-only graph (Init→Analyze→Research→Plan)."""
    builder = build_research_graph()
    with SqliteSaver.from_conn_string(config.CHECKPOINT_DB) as checkpointer:
        yield builder.compile(checkpointer=checkpointer)
