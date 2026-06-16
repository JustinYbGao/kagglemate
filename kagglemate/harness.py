"""Agent Harness / Agent 安全护栏

The Harness sits between the LLM and tool execution — the model CANNOT bypass it.
It provides: safety gates, audit logging, cost control, and operation risk classification.

与 System Prompt 的本质区别：
- System Prompt 是"建议"——模型可以选择忽略
- Harness 是"硬代码"——无论模型输出什么，都会在代码层被拦截

Architecture:
    LLM tool_call → Harness.pre_check() → [GO/STOP/MODIFY] → ToolExecutor → Harness.post_check() → audit log

Resume bullet:
    "Built multi-layer agent safety harness with operation risk classification,
     hard confirmation gates for destructive actions, immutable audit logging,
     and token budget enforcement — ensuring LLM cannot bypass safety constraints."
"""

from __future__ import annotations

import json
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional, Callable

from kagglemate.config import config


# ═══════════════════════════════════════════════════════════════════════════════
# Risk Level Classification / 操作风险等级
# ═══════════════════════════════════════════════════════════════════════════════

class RiskLevel:
    """Risk classification for tool operations.

    The Harness uses these to decide whether to block, confirm, or allow.
    """
    SAFE = "safe"            # Read-only, no side effects. Always allowed.
    READ_ONLY = "read_only"  # Reads data but may consume API quota.
    SIDE_EFFECT = "side_effect"   # Modifies local state (writes files, DB).
    DANGEROUS = "dangerous"  # Consumes limited resources (Kaggle submissions).
    CRITICAL = "critical"    # Could cause financial or reputational harm.
                             # ALWAYS requires human confirmation.


# Tool → Risk level mapping
TOOL_RISK_LEVELS = {
    "list_competitions": RiskLevel.SAFE,
    "research_competition": RiskLevel.READ_ONLY,       # downloads data (quota-expensive)
    "inspect_competition": RiskLevel.SAFE,
    "generate_baseline": RiskLevel.SIDE_EFFECT,         # writes script files
    "run_experiment": RiskLevel.SIDE_EFFECT,             # writes submission.csv
    "tune_model": RiskLevel.SIDE_EFFECT,                 # writes script + runs Optuna
    "get_suggestions": RiskLevel.READ_ONLY,
    "list_experiments": RiskLevel.SAFE,
    "show_experiment": RiskLevel.SAFE,
    "record_lb_score": RiskLevel.SIDE_EFFECT,            # modifies DB
    "ensemble_blend": RiskLevel.SIDE_EFFECT,             # writes submission file
    "validate_submission": RiskLevel.SAFE,
    "submit_to_kaggle": RiskLevel.DANGEROUS,             # consumes daily Kaggle slot
    "check_submission_status": RiskLevel.SAFE,
    "pull_notebook": RiskLevel.READ_ONLY,
    "read_generated_file": RiskLevel.SAFE,
}


# ═══════════════════════════════════════════════════════════════════════════════
# Session Budget / 会话预算
# ═══════════════════════════════════════════════════════════════════════════════

class SessionBudget:
    """Track and enforce per-session resource limits.

    Prevents runaway agent loops from consuming excessive tokens or API calls.
    """

    def __init__(self,
                 max_tool_calls: int = 50,
                 max_dangerous_ops: int = 3,
                 max_api_tokens_est: int = 500_000):
        self.max_tool_calls = max_tool_calls
        self.max_dangerous_ops = max_dangerous_ops
        self.max_api_tokens_est = max_api_tokens_est

        self.tool_calls_made = 0
        self.dangerous_ops_made = 0
        self.api_calls_made = 0
        self.estimated_tokens_spent = 0
        self.session_start = datetime.now(timezone.utc)
        self.warnings_issued = 0

    def record_api_call(self, prompt_tokens: int = 0, completion_tokens: int = 0):
        """Record an API call for budget tracking."""
        self.api_calls_made += 1
        self.estimated_tokens_spent += prompt_tokens + completion_tokens

    def check(self, risk_level: str) -> tuple[bool, str]:
        """Check if the operation is within budget.

        Returns (allowed, reason).
        """
        self.tool_calls_made += 1

        if self.tool_calls_made > self.max_tool_calls:
            return False, (
                f"会话工具调用次数已达上限 ({self.max_tool_calls})。"
                f"请重启会话继续。"
            )

        if risk_level in (RiskLevel.DANGEROUS, RiskLevel.CRITICAL):
            self.dangerous_ops_made += 1
            if self.dangerous_ops_made > self.max_dangerous_ops:
                return False, (
                    f"危险操作次数已达上限 ({self.max_dangerous_ops})。"
                    f"本次会话已执行 {self.dangerous_ops_made} 次危险操作。"
                )

        if self.estimated_tokens_spent > self.max_api_tokens_est:
            return False, (
                f"预估 Token 消耗已达上限 (~{self.max_api_tokens_est:,})。"
                f"请重启会话继续。"
            )

        # Warning at 80%
        if self.tool_calls_made > self.max_tool_calls * 0.8 and self.warnings_issued == 0:
            self.warnings_issued += 1
            return True, (
                f"⚠️ 已使用 {self.tool_calls_made}/{self.max_tool_calls} 次工具调用。"
            )

        return True, ""

    def summary(self) -> str:
        elapsed = (datetime.now(timezone.utc) - self.session_start).total_seconds()
        return (
            f"工具调用: {self.tool_calls_made}/{self.max_tool_calls} | "
            f"危险操作: {self.dangerous_ops_made}/{self.max_dangerous_ops} | "
            f"API 调用: {self.api_calls_made} | "
            f"预估 Token: ~{self.estimated_tokens_spent:,} | "
            f"运行时间: {elapsed:.0f}s"
        )


# ═══════════════════════════════════════════════════════════════════════════════
# Audit Trail / 审计日志
# ═══════════════════════════════════════════════════════════════════════════════

class AuditTrail:
    """Immutable audit log of every agent action.

    Written to ~/.kagglemate/audit.log — survives session restarts.
    Each entry is a JSON line with timestamp, action, risk_level, result summary.
    """

    def __init__(self, log_dir: Optional[Path] = None):
        self.log_dir = log_dir or (Path.home() / ".kagglemate")
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.log_dir / "audit.log"

    def record(self, tool_name: str, args: dict, risk_level: str,
               result_summary: str, success: bool, duration_ms: float,
               blocked: bool = False, blocked_reason: str = ""):
        """Append an immutable audit entry."""
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "tool": tool_name,
            "args_summary": _summarize_args(tool_name, args),
            "risk_level": risk_level,
            "success": success,
            "duration_ms": round(duration_ms, 1),
            "result_summary": result_summary[:200],
            "blocked": blocked,
            "blocked_reason": blocked_reason[:200],
        }
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    def recent(self, n: int = 20) -> list[dict]:
        """Read the most recent audit entries."""
        if not self.log_path.exists():
            return []
        lines = []
        with open(self.log_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    lines.append(json.loads(line))
        return lines[-n:]

    def count(self) -> int:
        """Total number of audit entries."""
        if not self.log_path.exists():
            return 0
        with open(self.log_path, "r", encoding="utf-8") as f:
            return sum(1 for _ in f)


def _summarize_args(tool_name: str, args: dict) -> str:
    """Create a short, safe summary of tool arguments (no secrets)."""
    if "api_key" in args:
        return "{...api_key redacted...}"
    # For most tools, just show the competition slug + key arg
    slug = args.get("competition_slug", "")
    if slug:
        others = {k: v for k, v in args.items() if k != "competition_slug"}
        others_str = ", ".join(f"{k}={v}" for k, v in list(others.items())[:3])
        return f"slug={slug}" + (f", {others_str}" if others_str else "")
    return str(args)[:100]


# ═══════════════════════════════════════════════════════════════════════════════
# Confirmation Gate / 确认门控
# ═══════════════════════════════════════════════════════════════════════════════

class ConfirmationGate:
    """Hard confirmation gate for dangerous operations.

    This is NOT a system prompt suggestion — it's a code-level barrier.
    The LLM CANNOT bypass it. Even if the model outputs a tool call,
    the Harness intercepts and requires human confirmation before execution.

    Architecture:
        1. Agent wants to execute DANGEROUS tool
        2. Harness checks → RiskLevel.DANGEROUS → requires confirmation
        3. User must explicitly approve (type 'YES')
        4. Only then does execution proceed
        5. Audit log records: who approved, when, what was executed
    """

    def __init__(self, io_handler: Callable[[str], str]):
        """io_handler: function to get user input. Typically input()."""
        self.io_handler = io_handler
        self.confirmed_actions: set[str] = set()  # one-shot confirmations
        self.super_confirm_mode = False  # if user says "yes to all"

    def require_confirmation(self, tool_name: str, args: dict,
                             risk_level: str,
                             context: str = "") -> tuple[bool, str]:
        """Ask user to confirm a dangerous operation.

        Returns (approved, reason).

        If user is in "super confirm" mode, auto-approve non-critical ops.
        """
        # Super confirm mode: auto-approve DANGEROUS, still block CRITICAL
        if self.super_confirm_mode and risk_level == RiskLevel.DANGEROUS:
            return True, "super_confirm_mode"

        # CRITICAL operations always require confirmation, even in super mode
        # (currently no CRITICAL ops defined, but this is future-proof)

        # Check if this exact action was already confirmed (one-shot)
        action_key = f"{tool_name}:{json.dumps(args, sort_keys=True)}"
        if action_key in self.confirmed_actions:
            return True, "previously_confirmed"

        # Build confirmation prompt
        prompt = _confirmation_prompt(tool_name, args, risk_level, context)

        # Get user input
        try:
            user_input = self.io_handler(prompt)
        except (EOFError, KeyboardInterrupt):
            return False, "用户取消 (EOF/Interrupt)"

        user_input = user_input.strip().upper()

        if user_input == "YES":
            self.confirmed_actions.add(action_key)
            return True, "user_confirmed"
        elif user_input == "YES TO ALL":
            self.super_confirm_mode = True
            self.confirmed_actions.add(action_key)
            return True, "user_confirmed_all"
        else:
            return False, f"用户拒绝 (输入: '{user_input[:20]}')"


def _confirmation_prompt(tool_name: str, args: dict, risk_level: str, context: str) -> str:
    """Format a confirmation prompt for the user."""
    icon = "🔴" if risk_level == RiskLevel.CRITICAL else "🟡"
    parts = [
        "",
        f"{icon} 操作需要确认 / Confirmation Required",
        f"   操作: {tool_name}",
    ]
    slug = args.get("competition_slug", "")
    if slug:
        parts.append(f"   比赛: {slug}")
    if tool_name == "submit_to_kaggle":
        parts.append(f"   文件: {args.get('file_path', '?')}")
        parts.append(f"   ⚠️  将消耗一次 Kaggle 每日提交机会")
    if context:
        parts.append(f"   {context}")
    parts.append("")
    parts.append("   输入 YES 确认，输入其他取消 / Type YES to confirm:")
    return "\n".join(parts)


# ═══════════════════════════════════════════════════════════════════════════════
# Harness — Main Orchestrator / 主协调器
# ═══════════════════════════════════════════════════════════════════════════════

class Harness:
    """The main Agent Harness.

    Wraps a ToolExecutor and enforces:
    1. Risk-level gating (DANGEROUS ops require confirmation)
    2. Budget enforcement (max tool calls, max dangerous ops, token budget)
    3. Audit logging (every action, blocked or executed, is recorded)
    4. Pre/post execution hooks for future extensibility

    Usage:
        harness = Harness(executor, io_handler=input)
        approved, result = harness.execute("submit_to_kaggle", {...})
    """

    def __init__(self, executor, io_handler=None, budget: Optional[SessionBudget] = None):
        self.executor = executor
        self.io_handler = io_handler or input
        self.budget = budget or SessionBudget()
        self.audit = AuditTrail()
        self.confirmation_gate = ConfirmationGate(self.io_handler)

        # ── Pre-execution hooks (extensible) ──
        self.pre_hooks: list[Callable] = []
        # ── Post-execution hooks (extensible) ──
        self.post_hooks: list[Callable] = []

        # ── Override: user can temporarily disable specific checks ──
        # (Only settable by the human, not by the LLM)
        self.confirmation_required = True

    def execute(self, tool_name: str, args: dict) -> tuple[bool, str]:
        """Execute a tool through the full safety pipeline.

        Returns (success, result_string).
        The result_string is either the tool output OR a harness rejection message.
        """
        risk_level = TOOL_RISK_LEVELS.get(tool_name, RiskLevel.READ_ONLY)

        t_start = time.time()

        # ════════════════ PHASE 1: Budget Check / 预算检查 ════════════════
        allowed, budget_msg = self.budget.check(risk_level)
        if not allowed:
            self.audit.record(tool_name, args, risk_level,
                              result_summary=budget_msg,
                              success=False, duration_ms=0,
                              blocked=True, blocked_reason=budget_msg)
            return False, f"🛑 [Harness] {budget_msg}"

        # ════════════════ PHASE 2: Risk Gate / 风险门控 ════════════════
        if self.confirmation_required and risk_level in (RiskLevel.DANGEROUS, RiskLevel.CRITICAL):
            context = ""
            if tool_name == "submit_to_kaggle":
                context = f"实验 #{args.get('experiment_id', '?')} 的提交文件"

            approved, reason = self.confirmation_gate.require_confirmation(
                tool_name, args, risk_level, context
            )
            if not approved:
                block_msg = f"⛔ [Harness] 操作被拦截: {tool_name} ({reason})"
                self.audit.record(tool_name, args, risk_level,
                                  result_summary=block_msg,
                                  success=False, duration_ms=0,
                                  blocked=True, blocked_reason=reason)
                return False, block_msg

        # ════════════════ PHASE 3: Pre-hooks / 前置钩子 ════════════════
        for hook in self.pre_hooks:
            try:
                hook(tool_name, args, risk_level)
            except Exception as e:
                self.audit.record(tool_name, args, risk_level,
                                  result_summary=f"Pre-hook failed: {e}",
                                  success=False, duration_ms=0,
                                  blocked=True, blocked_reason=str(e))
                return False, f"🛑 [Harness] Pre-hook rejected: {e}"

        # ════════════════ PHASE 4: Execute / 执行 ════════════════
        try:
            result = self.executor.execute(tool_name, args)
            success = True
        except Exception as e:
            result = f"Tool execution error: {e}"
            success = False

        duration_ms = (time.time() - t_start) * 1000

        # ════════════════ PHASE 5: Post-hooks / 后置钩子 ════════════════
        for hook in self.post_hooks:
            try:
                hook(tool_name, args, risk_level, result, success, duration_ms)
            except Exception:
                pass  # post-hooks never block

        # ════════════════ PHASE 6: Audit / 审计记录 ════════════════
        self.audit.record(
            tool_name=tool_name,
            args=args,
            risk_level=risk_level,
            result_summary=result[:200] if success else str(result)[:200],
            success=success,
            duration_ms=duration_ms,
            blocked=False,
        )

        return success, result

    def add_pre_hook(self, hook: Callable):
        """Add a pre-execution hook. Hook receives (tool_name, args, risk_level).
        Raise an exception to block execution."""
        self.pre_hooks.append(hook)

    def add_post_hook(self, hook: Callable):
        """Add a post-execution hook. Errors are silently caught."""
        self.post_hooks.append(hook)

    def status(self) -> str:
        """Get a human-readable harness status report."""
        lines = [
            "═══════════════════════════════════════",
            "  Harness Status / 护栏状态",
            "═══════════════════════════════════════",
            f"  审计日志: {self.audit.count()} 条记录",
            f"  审计文件: {self.audit.log_path}",
            f"  会话预算: {self.budget.summary()}",
            f"  确认门控: {'启用' if self.confirmation_required else '禁用'}",
            f"  超级确认: {'开启' if self.confirmation_gate.super_confirm_mode else '关闭'}",
            f"  预执行钩子: {len(self.pre_hooks)} 个",
            f"  后执行钩子: {len(self.post_hooks)} 个",
            "═══════════════════════════════════════",
        ]
        return "\n".join(lines)
