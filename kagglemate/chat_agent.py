"""Conversational Agent — talk to KaggleMate in natural language.

Architecture:
    You (自然语言) → DeepSeek V4 Pro → tool calls → execute nodes → response → 你

The 13 LangGraph nodes are exposed as callable tools.
DeepSeek acts as the "brain": understands intent, calls tools, synthesizes responses.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path
from typing import Optional

from openai import OpenAI
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.live import Live
from rich.spinner import Spinner

from kagglemate.config import config
from kagglemate.graph.state import KaggleAgentState
from kagglemate.tools.kaggle_cli import KaggleCLI

console = Console()

# ── System Prompt ──

SYSTEM_PROMPT = """You are KaggleMate, a Kaggle competition assistant. You help users research competitions, build models, track experiments, and submit predictions. You are proactive, helpful, and communicate in the user's language.

## Your Capabilities
- Research any Kaggle competition: download data, profile it, analyze public notebooks, generate strategy documents
- Generate and run baseline ML models (LightGBM, XGBoost, CatBoost)
- Tune hyperparameters with Optuna
- Blend multiple submissions into ensembles
- Track experiments in a database
- Validate submissions before uploading
- Submit predictions to Kaggle (the user has Kaggle API credentials configured)
- Check submission status and leaderboard scores
- Pull public notebooks from Kaggle

## How to Behave
1. **Be conversational** — speak naturally, like a teammate. If the user speaks Chinese, respond in Chinese.
2. **Be proactive** — after completing a task, suggest the logical next step.
3. **Explain your actions** — before running a slow task (research, training, tuning), tell the user what you're about to do.
4. **NEVER submit to Kaggle without explicit user confirmation.** Submissions consume daily quota.
5. **Track context** — remember the current competition the user is working on.
6. **When uncertain**, ask the user rather than guessing.

## Competition Context
The user is working on competitions. The most common workflow is:
1. Research a competition → 2. Generate baseline → 3. Run it → 4. Get suggestions → 5. Iterate → 6. Submit

Guide the user through this flow naturally.
"""

# ── Tool Definitions ──

TOOLS = [
    {
        "type": "function",
        "function": {
            "name": "list_competitions",
            "description": "列出当前活跃的 Kaggle 比赛（按最新发布排序）/ List ACTIVE Kaggle competitions sorted by newest first. Use when the user wants to browse or find a competition to enter.",
            "parameters": {
                "type": "object",
                "properties": {
                    "search": {
                        "type": "string",
                        "description": "搜索词，如 'tabular', 'playground', 'NLP' / Search term"
                    },
                    "category": {
                        "type": "string",
                        "description": "比赛类别: all, featured, research, playground, gettingStarted, recruitment",
                        "enum": ["all", "featured", "research", "playground", "gettingStarted", "recruitment"]
                    }
                },
                "required": []
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "research_competition",
            "description": "研究一个 Kaggle 比赛：下载数据、分析数据结构、调研公开 Notebook、生成策略文档(SPEC.md)。/ Research a competition: download data, profile, analyze notebooks, generate SPEC.md. This takes 1-3 minutes.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {
                        "type": "string",
                        "description": "比赛标识，如 'titanic', 'playground-series-s5e6' / Competition slug"
                    }
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "inspect_competition",
            "description": "查看比赛数据文件但不下载 / Inspect competition files without downloading. Use when user wants a quick look.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"}
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "generate_baseline",
            "description": "为比赛生成 baseline 训练脚本 / Generate a baseline ML training script for a competition. Requires research to be completed first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"}
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "run_experiment",
            "description": "执行训练脚本，记录 CV 分数和实验结果 / Run a training script, record CV score and experiment results. Requires baseline or tune to be generated first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "script_path": {
                        "type": "string",
                        "description": "训练脚本路径，留空则自动选择最新的 / Path to script, auto-selects latest if empty"
                    }
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "tune_model",
            "description": "用 Optuna 自动调参 / Hyperparameter tuning with Optuna. Generates and optionally runs tuning script.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "trials": {
                        "type": "integer",
                        "description": "Optuna 试验次数，默认 30 / Number of trials, default 30"
                    },
                    "run_immediately": {
                        "type": "boolean",
                        "description": "是否生成后立即运行 / Whether to run immediately after generating"
                    }
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "get_suggestions",
            "description": "根据实验历史给出下一步优化建议 / Get next-step recommendations based on experiment history.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"}
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_experiments",
            "description": "查看实验记录 / View experiment history for a competition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"}
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "show_experiment",
            "description": "查看某个实验的详细信息 / Show detailed info about a specific experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "experiment_id": {"type": "integer", "description": "实验 ID / Experiment ID"}
                },
                "required": ["competition_slug", "experiment_id"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "record_lb_score",
            "description": "手动记录排行榜(LB)分数到某个实验 / Record a leaderboard (LB) score for an experiment.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "experiment_id": {"type": "integer", "description": "实验 ID / Experiment ID"},
                    "lb_score": {"type": "number", "description": "排行榜分数 / Leaderboard score"}
                },
                "required": ["competition_slug", "experiment_id", "lb_score"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "ensemble_blend",
            "description": "将多个实验的提交文件融合 / Blend multiple experiment submissions into one ensemble.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "experiment_ids": {
                        "type": "string",
                        "description": "逗号分隔的实验 ID，如 '1,2,3' / Comma-separated experiment IDs"
                    },
                    "method": {
                        "type": "string",
                        "description": "融合方法: simple_average, weighted_average (推荐), rank_average",
                        "enum": ["simple_average", "weighted_average", "rank_average"]
                    }
                },
                "required": ["competition_slug", "experiment_ids"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "validate_submission",
            "description": "验证提交文件格式是否正确 / Validate a submission file format before submitting. Always run this before submitting.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "file_path": {"type": "string", "description": "提交文件路径 / Path to submission CSV"}
                },
                "required": ["competition_slug", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "submit_to_kaggle",
            "description": "提交预测文件到 Kaggle / Submit predictions to Kaggle. ⚠️ IMPORTANT: This consumes a daily submission slot. ONLY call this when the user explicitly asks to submit. Always validate first.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"},
                    "file_path": {"type": "string", "description": "提交文件路径 / Path to submission CSV"},
                    "message": {"type": "string", "description": "提交备注 / Submission message"}
                },
                "required": ["competition_slug", "file_path"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "check_submission_status",
            "description": "查看比赛提交历史和榜单状态 / Check submission history and leaderboard status for a competition.",
            "parameters": {
                "type": "object",
                "properties": {
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"}
                },
                "required": ["competition_slug"]
            }
        }
    },
    {
        "type": "function",
        "function": {
            "name": "pull_notebook",
            "description": "从 Kaggle 拉取公开 Notebook（保留依赖信息）/ Pull a public Kaggle notebook with metadata preserved.",
            "parameters": {
                "type": "object",
                "properties": {
                    "kernel_ref": {
                        "type": "string",
                        "description": "Notebook 引用，如 'username/notebook-name' / Kernel reference"
                    },
                    "competition_slug": {"type": "string", "description": "比赛标识 / Competition slug"}
                },
                "required": ["kernel_ref", "competition_slug"]
            }
        }
    },
]

# ── Tool Implementation ──


class ToolExecutor:
    """Maps DeepSeek tool calls to kagglemate node invocations."""

    def __init__(self):
        self.current_competition: Optional[str] = None

    def execute(self, tool_name: str, arguments: dict) -> str:
        """Execute a tool and return the result as a string."""
        method = getattr(self, f"_tool_{tool_name}", None)
        if method is None:
            return f"Unknown tool: {tool_name}"
        try:
            return method(arguments)
        except Exception as e:
            return f"Error executing {tool_name}: {e}"

    def _tool_list_competitions(self, args: dict) -> str:
        search = args.get("search", "")
        category = args.get("category", "all")

        comps = KaggleCLI.list_competitions(search=search, category=category, sort_by="recentlyCreated")

        if not comps:
            return f"没有找到比赛（搜索: '{search or 'all'}'）。Kaggle API 返回的比赛有限，建议直接访问 kaggle.com/competitions 浏览。"

        # Split into active (2025+) and old
        active = [c for c in comps if c.get("deadline", "") >= "2025"]
        old = [c for c in comps if c.get("deadline", "") < "2025"]

        lines = [f"## 活跃比赛 ({len(active)} 个)  / Active Competitions\n"]
        for c in active:
            ref = c.get("ref", "?").split("/")[-1]
            title = c.get("title", ref)
            deadline = (c.get("deadline", "") or "")[:10]
            lines.append(
                f"- **{ref}**: {title}\n"
                f"  截止: {deadline} | 类别: {c.get('category', '?')} | "
                f"奖励: {c.get('reward', '?')} | 队伍: {c.get('teamCount', '?')}"
            )

        if not active:
            lines.append("⚠️ Kaggle API 未返回活跃比赛。这**不代表没有**——Kaggle API 的列表功能很有限。")
            lines.append("建议直接访问 https://www.kaggle.com/competitions 查看完整列表。")
            lines.append(f"\n如果你已经知道比赛名称（如 playground-series-s5e6），直接告诉我就行，我可以直接研究它。")

        if old and len(active) < 10:
            lines.append(f"\n## 往期比赛 (前 {min(len(old), 5)} 个)")
            for c in old[:5]:
                ref = c.get("ref", "?").split("/")[-1]
                lines.append(f"- {ref} (截止: {(c.get('deadline','') or '')[:10]})")

        lines.append(f"\n💡 **提示**: 如果你已经知道比赛 slug（比如从 Kaggle 网页上看到的），直接告诉我就行，不需要通过列表查找。")
        return "\n".join(lines)

    def _tool_research_competition(self, args: dict) -> str:
        slug = args["competition_slug"]
        self.current_competition = slug

        from kagglemate.graph.nodes.init_node import run as init_run
        from kagglemate.graph.nodes.analyze_node import run as analyze_run
        from kagglemate.graph.nodes.research_node import run as research_run
        from kagglemate.graph.nodes.plan_node import run as plan_run

        state: KaggleAgentState = _base_state(slug)

        # Init: download data
        console.print("  [dim]⏳ 下载数据中...[/]")
        state.update(init_run(state))
        if state.get("errors"):
            return f"下载失败: {state['errors'][0]}"

        # Analyze: profile data
        console.print("  [dim]⏳ 分析数据中...[/]")
        state.update(analyze_run(state))

        # Research: find notebooks
        console.print("  [dim]⏳ 调研公开 Notebook 中...[/]")
        state.update(research_run(state))

        # Plan: generate SPEC
        console.print("  [dim]⏳ 生成策略文档中...[/]")
        state.update(plan_run(state))

        ctype = state.get("competition_type", "?")
        metric = state.get("evaluation_metric", "?")
        profile = state.get("data_profile") or {}
        n_nb = len(state.get("notebook_summaries", []))
        report_dir = state.get("report_dir", "")

        summary = f"""## 调研完成 — {slug}

**任务类型**: {ctype}
**评价指标**: {metric}
**训练集**: {profile.get('train_rows', '?')} 行 × {len(profile.get('columns', []))} 列
**测试集**: {profile.get('test_rows', '?')} 行
**目标变量**: `{profile.get('target_col', '?')}`
**公开 Notebook**: {n_nb} 个

已生成文件：
- `SPEC.md` — 完整竞赛策略文档
- `data_profile.md` — 数据分析报告
- `rules_checklist.md` — 规则检查清单

**下一步**: 输入 "生成 baseline" 来创建第一个模型。"""

        if n_nb == 0:
            summary += "\n\n⚠️ 该比赛暂无公开 Notebook（可能是新比赛），策略文档基于数据特征生成。"

        return summary

    def _tool_inspect_competition(self, args: dict) -> str:
        slug = args["competition_slug"]
        self.current_competition = slug
        try:
            files = KaggleCLI.list_files(slug)
        except Exception as e:
            return f"无法获取文件: {e}"

        if not files:
            return f"比赛 `{slug}` 暂无可用文件（可能需要先接受规则）。"

        lines = [f"**{slug}** 的数据文件："]
        for f in files:
            lines.append(f"- `{f.get('name', '?')}` ({f.get('size', '?')} bytes)")
        return "\n".join(lines)

    def _tool_generate_baseline(self, args: dict) -> str:
        slug = args["competition_slug"]
        _ensure_data(slug)

        from kagglemate.graph.nodes.analyze_node import run as analyze_run
        from kagglemate.graph.nodes.baseline_node import run as baseline_run

        state: KaggleAgentState = _base_state(slug)
        state["data_dir"] = str(config.COMPETITIONS_DIR / slug / "data" / "raw")
        state["report_dir"] = str(config.COMPETITIONS_DIR / slug / "reports")
        state["script_dir"] = str(config.COMPETITIONS_DIR / slug / "scripts")
        state["submission_dir"] = str(config.COMPETITIONS_DIR / slug / "submissions")

        state.update(analyze_run(state))
        state.update(baseline_run(state))

        exp = state.get("current_experiment") or {}
        script = exp.get("script_path", "")
        model = exp.get("model", "?")
        n_features = len(exp.get("features", []))

        return f"""## Baseline 脚本已生成

**模型**: {model}
**特征数**: {n_features}
**脚本路径**: `{script}`

**下一步**: 输入 "运行 baseline" 来执行训练。"""

    def _tool_run_experiment(self, args: dict) -> str:
        slug = args["competition_slug"]
        script_path = args.get("script_path", "")

        from kagglemate.graph.nodes.run_node import run as run_node_fn

        # Auto-find script
        if not script_path:
            scripts_dir = config.COMPETITIONS_DIR / slug / "scripts"
            if scripts_dir.exists():
                candidates = sorted(scripts_dir.glob("train_baseline_*.py"), reverse=True) + \
                             sorted(scripts_dir.glob("tune_*.py"), reverse=True)
                if candidates:
                    script_path = str(candidates[0])

        if not script_path or not Path(script_path).exists():
            return "未找到训练脚本。请先生成 baseline：输入 '生成 baseline'。"

        state: KaggleAgentState = _base_state(slug)
        state["data_dir"] = str(config.COMPETITIONS_DIR / slug / "data" / "raw")
        state["submission_dir"] = str(config.COMPETITIONS_DIR / slug / "submissions")
        state["script_dir"] = str(config.COMPETITIONS_DIR / slug / "scripts")
        state["current_experiment"] = {
            "name": Path(script_path).stem,
            "model": "Unknown",
            "script_path": script_path,
            "status": "running",
        }

        console.print("  [dim]⏳ 训练中...[/]")
        state.update(run_node_fn(state))

        exp = state.get("current_experiment") or {}
        errors = state.get("errors", [])

        if errors and exp.get("status") == "failed":
            return f"训练失败: {errors[0][:500]}"

        cv = exp.get("cv_score", 0.0)
        cv_std = exp.get("cv_std", 0.0)
        sub = exp.get("submission_path", "")
        exp_id = exp.get("id", "?")

        # Get best comparison
        from kagglemate.memory.experiment_store import ExperimentStore
        store = ExperimentStore(slug)
        best = store.get_best()
        comp_text = ""
        if best and best["id"] != exp_id:
            best_cv = best.get("cv_score", 0)
            if cv > best_cv:
                comp_text = f"\n🏆 **新的最佳分数！** (之前: {best_cv:.5f})"
            else:
                comp_text = f"\n📊 当前最佳: {best_cv:.5f} (差距: {best_cv - cv:.5f})"

        return f"""## 实验 #{exp_id} 完成

**CV 分数**: {cv:.5f} ± {cv_std:.5f}
**提交文件**: `{sub}`{comp_text}

**下一步**: 输入 "给建议" 获取优化方向，或 "调参" 进行超参数优化。"""

    def _tool_tune_model(self, args: dict) -> str:
        slug = args["competition_slug"]
        trials = args.get("trials", 30)
        run_now = args.get("run_immediately", False)

        _ensure_data(slug)

        from kagglemate.graph.nodes.analyze_node import run as analyze_run
        from kagglemate.graph.nodes.tune_node import run as tune_run

        state: KaggleAgentState = _base_state(slug)
        state["data_dir"] = str(config.COMPETITIONS_DIR / slug / "data" / "raw")
        state["report_dir"] = str(config.COMPETITIONS_DIR / slug / "reports")
        state["script_dir"] = str(config.COMPETITIONS_DIR / slug / "scripts")
        state["submission_dir"] = str(config.COMPETITIONS_DIR / slug / "submissions")
        state["tune_trials"] = trials

        state.update(analyze_run(state))
        state.update(tune_run(state))

        exp = state.get("current_experiment") or {}
        script = exp.get("script_path", "")

        if run_now:
            console.print(f"  [dim]⏳ 运行 {trials} 次 Optuna 试验中...[/]")
            from kagglemate.graph.nodes.run_node import run as run_node_fn
            run_state = _base_state(slug)
            run_state.update(state)
            run_state["current_experiment"] = exp
            run_state["script_dir"] = str(config.COMPETITIONS_DIR / slug / "scripts")
            run_state["data_dir"] = str(config.COMPETITIONS_DIR / slug / "data" / "raw")
            run_state["submission_dir"] = str(config.COMPETITIONS_DIR / slug / "submissions")
            run_state.update(run_node_fn(run_state))

            run_exp = run_state.get("current_experiment") or {}
            cv = run_exp.get("cv_score", 0.0)
            exp_id = run_exp.get("id", "?")

            return f"""## 调参完成

**试验次数**: {trials}
**最佳 CV**: {cv:.5f}
**实验 #{exp_id}** 已保存

**下一步**: 输入 "给建议" 看看还有什么优化方向。"""

        return f"""## 调参脚本已生成

**试验次数**: {trials}
**脚本路径**: `{script}`

**下一步**: 输入 "运行调参" 来执行优化（预计需要 {trials * 15} 秒到 {trials * 60} 秒）。"""

    def _tool_get_suggestions(self, args: dict) -> str:
        slug = args["competition_slug"]
        _ensure_data(slug)

        from kagglemate.graph.nodes.suggest_node import run as suggest_run

        state: KaggleAgentState = _base_state(slug)
        state["report_dir"] = str(config.COMPETITIONS_DIR / slug / "reports")
        state["data_dir"] = str(config.COMPETITIONS_DIR / slug / "data" / "raw")

        state.update(suggest_run(state))

        report_dir = config.COMPETITIONS_DIR / slug / "reports"
        next_steps = report_dir / "next_steps.md"
        if next_steps.exists():
            content = next_steps.read_text()
            # Return the recommendations section
            if "## Recommended Next Experiments / 推荐下一步实验" in content:
                return content.split("## Recommended Next Experiments / 推荐下一步实验")[1].split("---")[0]
            return content[-2000:]  # last 2000 chars
        return "无法生成建议。请先完成至少一次实验。"

    def _tool_list_experiments(self, args: dict) -> str:
        slug = args["competition_slug"]
        from kagglemate.memory.experiment_store import ExperimentStore

        store = ExperimentStore(slug)
        exps = store.list_all()

        if not exps:
            return f"`{slug}` 暂无实验记录。输入 '生成 baseline' 开始。"

        lines = [f"**{slug}** 的实验记录 ({len(exps)} 个)：\n"]
        lines.append("| # | 名称 | 模型 | CV | LB | 状态 |")
        lines.append("|---|---|---|---|---|---|")
        for e in exps[:15]:
            cv = f"{e.get('cv_score', 0):.5f}" if e.get('cv_score') else "—"
            lb = f"{e.get('lb_score', 0):.5f}" if e.get('lb_score') else "—"
            lines.append(
                f"| {e['id']} | {e.get('experiment_name', '?')[:20]} | "
                f"{e.get('model_name', '?')[:10]} | {cv} | {lb} | {e.get('status', '?')} |"
            )
        return "\n".join(lines)

    def _tool_show_experiment(self, args: dict) -> str:
        slug = args["competition_slug"]
        exp_id = args["experiment_id"]

        from kagglemate.memory.experiment_store import ExperimentStore
        store = ExperimentStore(slug)
        exp = store.get(exp_id)

        if not exp:
            return f"实验 #{exp_id} 不存在。"

        fi = exp.get("feature_importance") or []
        fi_text = ""
        if fi:
            fi_lines = ["\n**特征重要性 Top 5**:"]
            for name, imp in fi[:5]:
                fi_lines.append(f"  - {name}: {imp:.4f}")
            fi_text = "\n".join(fi_lines)

        return f"""## 实验 #{exp_id}: {exp.get('experiment_name', '?')}

**模型**: {exp.get('model_name', '?')}
**CV 分数**: {exp.get('cv_score', 'N/A')}
**LB 分数**: {exp.get('lb_score', 'N/A')}
**指标**: {exp.get('metric', '?')}
**特征数**: {len(exp.get('features') or [])}
**状态**: {exp.get('status', '?')}{fi_text}"""

    def _tool_record_lb_score(self, args: dict) -> str:
        slug = args["competition_slug"]
        exp_id = args["experiment_id"]
        lb_score = args["lb_score"]

        from kagglemate.memory.experiment_store import ExperimentStore
        store = ExperimentStore(slug)
        ok = store.update_lb(exp_id, lb_score)

        if ok:
            exp = store.get(exp_id)
            cv = exp.get("cv_score", 0) if exp else 0
            gap = cv - lb_score if cv else 0
            return f"✅ 实验 #{exp_id} 的 LB 分数已记录为 {lb_score:.5f}。CV/LB 差距: {gap:.5f}"
        return f"实验 #{exp_id} 不存在。"

    def _tool_ensemble_blend(self, args: dict) -> str:
        slug = args["competition_slug"]
        ids_str = args["experiment_ids"]
        method = args.get("method", "weighted_average")

        exp_ids = [int(x.strip()) for x in ids_str.split(",") if x.strip()]

        from kagglemate.graph.nodes.ensemble_node import run as ensemble_run

        state: KaggleAgentState = _base_state(slug)
        state["ensemble_exp_ids"] = exp_ids
        state["ensemble_method"] = method
        state["submission_dir"] = str(config.COMPETITIONS_DIR / slug / "submissions")
        state["data_dir"] = str(config.COMPETITIONS_DIR / slug / "data" / "raw")

        state.update(ensemble_run(state))

        errors = state.get("errors", [])
        if errors:
            return f"融合失败: {errors[0]}"

        exp = state.get("current_experiment") or {}
        sub_path = exp.get("submission_path", "")

        return f"""## Ensemble 融合完成

**方法**: {method}
**实验数**: {len(exp_ids)}
**提交文件**: `{sub_path}`

**下一步**: 输入 "验证 {sub_path}" 检查格式，然后我可以帮你提交。"""

    def _tool_validate_submission(self, args: dict) -> str:
        slug = args["competition_slug"]
        file_path = args["file_path"]

        from kagglemate.tools.submission_validator import validate
        data_dir = str(config.COMPETITIONS_DIR / slug / "data" / "raw")
        vr = validate(file_path, data_dir)

        lines = [f"**验证结果** ({'✅ 通过' if vr.is_valid else '❌ 未通过'}):\n"]
        for c in vr.checks:
            icon = "✅" if c.passed else "❌"
            lines.append(f"- {icon} {c.check}: {c.detail}")

        if vr.warnings:
            lines.append(f"\n⚠️ **警告**:")
            for w in vr.warnings:
                lines.append(f"  - {w}")

        if vr.errors:
            lines.append(f"\n❌ **错误**:")
            for e in vr.errors:
                lines.append(f"  - {e}")

        if vr.is_valid:
            lines.append(f"\n✅ 文件格式正确，可以提交。输入 '提交 {file_path} 到 {slug}' 进行提交。")

        return "\n".join(lines)

    def _tool_submit_to_kaggle(self, args: dict) -> str:
        slug = args["competition_slug"]
        file_path = args["file_path"]
        message = args.get("message", "KaggleMate submission")

        # Step 1: Always validate first
        from kagglemate.tools.submission_validator import validate
        data_dir = str(config.COMPETITIONS_DIR / slug / "data" / "raw")
        vr = validate(file_path, data_dir)

        if not vr.is_valid:
            errors_text = "\n".join(f"- {e}" for e in vr.errors)
            return f"""## ❌ 无法提交 — 验证未通过

{errors_text}

请修复以上问题后再试。"""

        # Step 2: Verify Kaggle credentials
        kaggle_json = Path.home() / ".kaggle" / "kaggle.json"
        if not kaggle_json.exists():
            return "❌ 未找到 Kaggle API 凭证。请确保 `~/.kaggle/kaggle.json` 存在。"

        # Step 3: Submit
        from kagglemate.tools.kaggle_cli import KaggleCLI
        try:
            result = KaggleCLI.submit(slug, Path(file_path), message)
        except RuntimeError as e:
            return f"## ❌ 提交失败\n\n```\n{e}\n```\n\n可能原因: 今日提交次数已用完、文件格式不对、比赛已结束。"

        # Step 4: Try to link to experiment
        exp_info = ""
        from kagglemate.memory.experiment_store import ExperimentStore
        store = ExperimentStore(slug)
        # Find the experiment that produced this submission
        exps = store.list_all(limit=5)
        for e in exps:
            if e.get("submission_path") == file_path or e.get("submission_path", "").endswith(Path(file_path).name):
                exp_info = f"\n关联实验: **#{e['id']}** (CV: {e.get('cv_score', 'N/A')})"

        return f"""## ✅ 提交成功！

**比赛**: {slug}
**文件**: {Path(file_path).name}
**说明**: {message}{exp_info}

```{result.get('stdout', 'OK')[:300]}```

⚠️ **提醒**:
- 等待 **4 小时以上** 让分数稳定再判断
- 早期分数通常虚高
- 输入 "查看 {slug} 提交状态" 来追踪

需要我帮你记录 LB 分数时，告诉我实验编号和分数。"""

    def _tool_check_submission_status(self, args: dict) -> str:
        slug = args["competition_slug"]
        from kagglemate.tools.kaggle_cli import KaggleCLI

        try:
            subs = KaggleCLI.submissions(slug)
        except Exception as e:
            return f"无法获取提交记录: {e}"

        if not subs:
            return f"比赛 `{slug}` 暂无提交记录。"

        lines = [f"**{slug}** 的提交记录：\n"]
        lines.append("| 日期 | 说明 | 分数 | 状态 |")
        lines.append("|---|---|---|---|")
        for s in subs[:10]:
            date = (s.get("date", "") or "")[:16]
            desc = (s.get("description", "") or "")[:30]
            score = s.get("publicScore", "—") or "pending"
            status = s.get("status", "?")
            lines.append(f"| {date} | {desc} | {score} | {status} |")

        return "\n".join(lines)

    def _tool_pull_notebook(self, args: dict) -> str:
        kernel_ref = args["kernel_ref"]
        slug = args["competition_slug"]

        from kagglemate.graph.nodes.kernel_node import _pull_kernel

        state: KaggleAgentState = _base_state(slug)
        state["kernel_ref"] = kernel_ref
        state["kernel_action"] = "pull"

        state.update(_pull_kernel(state))

        errors = state.get("errors", [])
        kernel_dir = state.get("kernel_dir", "")

        if errors:
            return f"拉取失败: {errors[0]}"

        return f"✅ Notebook `{kernel_ref}` 已保存到 `{kernel_dir}`"


# ── Helpers ──


def _base_state(slug: str) -> KaggleAgentState:
    return {
        "competition_slug": slug,
        "competition_name": slug,
        "messages": [],
        "current_phase": "init",
        "errors": [],
        "best_cv_score": 0.0,
        "best_lb_score": 0.0,
        "human_approval_required": False,
        "human_approved": False,
    }


def _ensure_data(slug: str):
    """Ensure competition data exists, download if not."""
    data_dir = config.COMPETITIONS_DIR / slug / "data" / "raw"
    if not data_dir.exists() or not list(data_dir.glob("*.csv")):
        console.print(f"  [dim]⏳ 首次使用 {slug}，正在下载数据...[/]")
        from kagglemate.graph.nodes.init_node import run as init_run
        state = _base_state(slug)
        init_run(state)


# ── Conversation Loop ──


def chat():
    """Start the conversational KaggleMate agent."""
    _print_welcome()

    client = OpenAI(
        api_key=config.DEEPSEEK_API_KEY,
        base_url=config.DEEPSEEK_BASE_URL,
    )

    executor = ToolExecutor()
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]

    while True:
        try:
            user_input = console.input("\n[bold green]你[/]: ")
        except (EOFError, KeyboardInterrupt):
            console.print("\n[dim]再见！[/]\n")
            break

        if not user_input.strip():
            continue

        if user_input.lower() in ("exit", "quit", "退出", "q"):
            console.print("[dim]再见！Good luck with Kaggle! 🚀[/]\n")
            break

        messages.append({"role": "user", "content": user_input})

        # Call DeepSeek
        console.print()  # spacing
        try:
            response = client.chat.completions.create(
                model=config.DEEPSEEK_MODEL,
                messages=messages,
                tools=TOOLS,
                tool_choice="auto",
                extra_body={"thinking": {"type": "disabled"}},
            )
        except Exception as e:
            console.print(f"[red]API 错误: {e}[/]")
            continue

        msg = response.choices[0].message

        # Handle tool calls
        if msg.tool_calls:
            # Build tool_calls for message history (DeepSeek-compatible format)
            tool_call_entries = []
            for tc in msg.tool_calls:
                tool_call_entries.append({
                    "id": tc.id,
                    "type": "function",
                    "function": {
                        "name": tc.function.name,
                        "arguments": tc.function.arguments,
                    }
                })

            # Append assistant message with tool calls
            messages.append({
                "role": "assistant",
                "content": msg.content,  # may be None
                "tool_calls": tool_call_entries,
            })

            for tc in msg.tool_calls:
                tool_name = tc.function.name
                try:
                    args = json.loads(tc.function.arguments)
                except json.JSONDecodeError:
                    args = {}

                # Show what the agent is doing
                action_desc = _action_description(tool_name, args)
                with Live(Spinner("dots", text=f"  {action_desc}"), console=console, transient=True):
                    result = executor.execute(tool_name, args)
                    time.sleep(0.3)  # brief pause so spinner is visible

                # Show result
                console.print(Panel(Markdown(result), title="KaggleMate", border_style="cyan"))

                # Append tool result to messages
                messages.append({
                    "role": "tool",
                    "tool_call_id": tc.id,
                    "content": result,
                })

            # Get final response from DeepSeek after tool execution
            try:
                final_response = client.chat.completions.create(
                    model=config.DEEPSEEK_MODEL,
                    messages=messages,
                    extra_body={"thinking": {"type": "disabled"}},
                )
                final_msg = final_response.choices[0].message
                reply = final_msg.content or getattr(final_msg, "reasoning_content", None) or ""
                if reply.strip():
                    console.print()
                    console.print(Panel(Markdown(reply), border_style="cyan"))
                    messages.append({"role": "assistant", "content": reply})
            except Exception as e:
                console.print(f"  [dim]({e})[/]")

        else:
            # No tool calls — just display the response
            reply = msg.content or getattr(msg, "reasoning_content", None) or ""
            if reply.strip():
                console.print(Panel(Markdown(reply), border_style="cyan"))
                messages.append({"role": "assistant", "content": reply})


def _action_description(tool_name: str, args: dict) -> str:
    """Human-readable action description for spinner."""
    slug = args.get("competition_slug", "")
    descriptions = {
        "list_competitions": "正在搜索比赛...",
        "research_competition": f"正在研究 {slug}（下载数据→分析→调研）...",
        "inspect_competition": f"查看 {slug} 的文件...",
        "generate_baseline": f"正在为 {slug} 生成 baseline...",
        "run_experiment": f"正在运行 {slug} 的训练脚本...",
        "tune_model": f"正在为 {slug} 生成调参脚本...",
        "get_suggestions": f"正在分析 {slug} 的实验历史...",
        "list_experiments": f"查询 {slug} 的实验记录...",
        "show_experiment": "获取实验详情...",
        "record_lb_score": "记录 LB 分数...",
        "ensemble_blend": "正在融合模型...",
        "validate_submission": "验证提交文件...",
        "submit_to_kaggle": f"正在提交到 Kaggle: {slug}...",
        "check_submission_status": f"查询 {slug} 的提交状态...",
        "pull_notebook": f"拉取 Notebook: {args.get('kernel_ref', '')}...",
    }
    return descriptions.get(tool_name, f"调用 {tool_name}...")


def _print_welcome():
    """Print welcome banner."""
    console.print()
    console.print(Panel(
        "[bold cyan]🏆 KaggleMate Agent[/]\n\n"
        "你的 Kaggle 竞赛搭档。用自然语言告诉我你想做什么：\n\n"
        "• \"帮我研究一下 titanic 比赛\"\n"
        "• \"生成一个 baseline\"\n"
        "• \"跑一下训练\"\n"
        "• \"现在最佳分数是多少？给点建议\"\n"
        "• \"用 Optuna 调参 50 次\"\n"
        "• \"融合实验 1 和实验 2\"\n\n"
        "[dim]输入 'exit' 退出 / Type 'exit' to quit[/]",
        title="欢迎 / Welcome",
        border_style="cyan",
    ))
