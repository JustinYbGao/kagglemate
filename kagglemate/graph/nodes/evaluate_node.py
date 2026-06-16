"""Evaluate Node — assess experiment results and diagnose issues.

Phase 3: full evaluation with CV/LB gap analysis, trend detection,
overfitting checks, and strategy quality scoring.

Drives the conditional edge: → "suggest" (continue) or → END (stop).
"""

from __future__ import annotations

from kagglemate.graph.state import KaggleAgentState
from kagglemate.memory.experiment_store import ExperimentStore


def run(state: KaggleAgentState) -> dict:
    """Evaluate the most recent experiment and produce a diagnosis.

    Returns a `should_continue` flag that the conditional edge uses:
    - True → route to "suggest" node
    - False → route to END
    """
    exp = state.get("current_experiment") or {}
    slug = state["competition_slug"]
    store = ExperimentStore(slug)

    cv_score = exp.get("cv_score", 0.0)
    cv_std = exp.get("cv_std", 0.0)
    all_exps = store.list_all()
    total_exps = len(all_exps)
    best_exp = store.get_best()
    best_cv = best_exp.get("cv_score", 0.0)
    best_lb = best_exp.get("lb_score") if best_exp else None

    diagnosis: list[str] = []
    should_continue = True

    # ── 1. Status check ──
    if exp.get("status") == "failed":
        error = exp.get("error_message", "Unknown error")[:200]
        diagnosis.append(f"❌ Experiment FAILED: {error}")
        # If it's the first failure, keep going; if too many failures, stop
        failed_count = sum(1 for e in all_exps if e.get("status") == "failed")
        if failed_count >= 3:
            diagnosis.append(f"🛑 {failed_count} consecutive failures — stopping.")
            should_continue = False
    elif cv_score > 0:
        diagnosis.append(f"✅ Completed: CV = {cv_score:.5f} ± {cv_std:.5f}")

    # ── 2. Best score check ──
    if cv_score > best_cv and cv_score > 0:
        improvement = cv_score - best_cv
        diagnosis.append(f"🏆 NEW BEST! +{improvement:.5f} over previous ({best_cv:.5f})")
    elif cv_score > 0 and total_exps > 1:
        gap = best_cv - cv_score
        diagnosis.append(f"📉 {gap:.5f} below current best ({best_cv:.5f})")

    # ── 3. CV variance check ──
    if cv_std > 0 and cv_score > 0:
        cv_pct = (cv_std / cv_score) * 100
        if cv_pct > 10:
            diagnosis.append(f"⚠️ High CV variance: {cv_pct:.1f}% of mean (overfitting or unstable data)")
        elif cv_pct < 1:
            diagnosis.append(f"✅ Very stable CV: ±{cv_std:.5f}")
        else:
            diagnosis.append(f"ℹ️ CV stability OK: ±{cv_std:.5f} ({cv_pct:.1f}%)")

    # ── 4. CV/LB gap (detect overfitting) ──
    cv_lb_gap = None
    if best_exp and best_lb is not None and best_cv > 0:
        cv_lb_gap = best_cv - best_lb
        if cv_lb_gap > 0.05 and best_cv > 0:
            diagnosis.append(
                f"🚨 Large CV/LB gap: CV={best_cv:.5f}, LB={best_lb:.5f} "
                f"(gap={cv_lb_gap:.5f}). Possible overfitting or test set shift."
            )
        elif cv_lb_gap > 0.01:
            diagnosis.append(
                f"⚠️ Moderate CV/LB gap: {cv_lb_gap:.5f}. "
                f"Monitor after more submissions."
            )
        else:
            diagnosis.append(f"✅ CV/LB gap tight: {cv_lb_gap:.5f}")

    # ── 5. Plateau detection ──
    if total_exps >= 5:
        recent_cv = [e.get("cv_score", 0) for e in all_exps[:5] if e.get("cv_score")]
        if recent_cv:
            max_recent = max(recent_cv)
            if max_recent <= best_cv and (best_cv - max_recent) < 0.005:
                diagnosis.append(
                    "📊 Plateau detected: last 5 experiments show no improvement. "
                    "Consider bigger changes (new feature engineering, different model family)."
                )

    # ── 6. Feature coverage ──
    if exp.get("features"):
        n_features = len(exp["features"])
        if n_features < 5:
            diagnosis.append(f"💡 Only {n_features} features used — room for feature engineering.")
        else:
            diagnosis.append(f"ℹ️ {n_features} features used.")

    # ── 7. Should we continue? ──
    if total_exps >= 20:
        diagnosis.append("🛑 20 experiments reached — consider reviewing and submitting.")
        # Don't force stop, but flag it

    # ── 8. Experiment count ──
    diagnosis.append(f"📊 Total experiments: {total_exps}")

    for d in diagnosis:
        _log(d)

    return {
        "best_cv_score": max(cv_score, best_cv),
        "best_lb_score": best_lb if best_lb else (state.get("best_lb_score", 0.0)),
        "all_experiments": all_exps,
        "current_phase": "evaluate",
        # Signal for conditional edge routing
        "_should_continue": should_continue,
    }


def _log(msg: str):
    print(f"  [evaluate] {msg}")
