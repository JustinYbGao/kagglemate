"""Competition Type Registry / 比赛类型注册表

One source of truth for:
1. What competition types exist
2. How to detect each type from file extensions
3. What capabilities (tools) are available for each type
4. Auto-detection logic

To ADD a new competition type: just add an entry to COMPETITION_TYPES below.
The Harness will automatically enforce capability gates for the new type.

Architecture:
    research detects type → stores in state → Harness pre-hook checks tools against registry
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from kagglemate.config import config


# ═══════════════════════════════════════════════════════════════════════════════
# Competition Type Definitions / 比赛类型定义
# ═══════════════════════════════════════════════════════════════════════════════

class CompetitionType:
    """A competition type with detection rules and capability matrix.

    To add a new type: add an entry to COMPETITION_TYPES dict.
    The Harness and research pipeline will pick it up automatically.
    """

    def __init__(self,
                 type_id: str,
                 name_zh: str,
                 name_en: str,
                 detection_extensions: list[str],
                 detection_requires: Optional[dict] = None,
                 can_baseline: bool = False,
                 can_tune: bool = False,
                 can_ensemble: bool = False,
                 can_submit: bool = True,
                 can_research: bool = True,
                 can_profile: bool = True,
                 description_zh: str = "",
                 description_en: str = "",
                 example_slugs: list[str] = None):
        self.type_id = type_id
        self.name_zh = name_zh
        self.name_en = name_en
        self.detection_extensions = detection_extensions  # e.g. ['.csv']
        self.detection_requires = detection_requires or {}  # e.g. {"min_csv_rows": 1}
        self.can_baseline = can_baseline
        self.can_tune = can_tune
        self.can_ensemble = can_ensemble
        self.can_submit = can_submit
        self.can_research = can_research
        self.can_profile = can_profile
        self.description_zh = description_zh
        self.description_en = description_en
        self.example_slugs = example_slugs or []

    @property
    def allowed_tools(self) -> dict[str, bool]:
        """Map of tool_name → allowed."""
        return {
            "research_competition": self.can_research,
            "generate_baseline": self.can_baseline,
            "run_experiment": self.can_baseline,  # depends on baseline
            "tune_model": self.can_tune,
            "ensemble_blend": self.can_ensemble,
            "submit_to_kaggle": self.can_submit,
            "get_suggestions": self.can_research,  # needs experiment history
            "list_experiments": True,  # always allowed
            "show_experiment": True,
            "record_lb_score": True,
            "validate_submission": self.can_submit,
            "check_submission_status": True,
            "pull_notebook": self.can_research,
            "read_generated_file": True,
            "list_competitions": True,
            "inspect_competition": True,
        }

    def can_use(self, tool_name: str) -> bool:
        """Check if a specific tool is allowed for this type."""
        return self.allowed_tools.get(tool_name, False)

    def blocked_reason(self, tool_name: str) -> str:
        """Human-readable explanation of why a tool is blocked."""
        tool_names_zh = {
            "generate_baseline": "自动生成 baseline",
            "run_experiment": "自动运行训练",
            "tune_model": "自动调参",
            "ensemble_blend": "模型集成",
        }
        tool_zh = tool_names_zh.get(tool_name, tool_name)

        if self.can_research:
            return (
                f"🚫 {tool_zh}不适用于「{self.name_zh}」类型比赛。\n\n"
                f"**为什么**: {self.description_zh}\n\n"
                f"**我能做什么**:\n"
                f"  • 分析数据结构和格式\n"
                f"  • 调研公开 Notebook 和讨论\n"
                f"  • 生成比赛策略文档 (SPEC.md)\n"
                f"  • 拉取高分 Notebook 供参考\n"
                f"  • 检查提交文件格式\n"
                f"  • 查看提交状态\n\n"
                f"**你需要**: 手动编写解决方案代码。告诉我你想怎么处理这个比赛。"
            )
        return f"🚫 此操作不适用于「{self.name_zh}」比赛类型。"


# ═══════════════════════════════════════════════════════════════════════════════
# Registry — All Known Competition Types
# ═══════════════════════════════════════════════════════════════════════════════
#
# To add a new type, copy an existing entry and modify:
#   1. type_id, name_zh, name_en
#   2. detection_extensions (file extensions that indicate this type)
#   3. detection_requires (extra conditions, e.g. min files)
#   4. can_* flags (what the agent can do for this type)
#   5. description (shown to user when tools are blocked)
# ═══════════════════════════════════════════════════════════════════════════════

COMPETITION_TYPES: dict[str, CompetitionType] = {
    # ── Tabular CSV (fully supported) ──
    "tabular": CompetitionType(
        type_id="tabular",
        name_zh="表格数据 (Tabular CSV)",
        name_en="Tabular CSV",
        detection_extensions=[".csv"],
        detection_requires={"min_csv_rows": 100},
        can_baseline=True,
        can_tune=True,
        can_ensemble=True,
        can_submit=True,
        description_zh="这是标准的表格数据竞赛。支持全自动 pipeline：LightGBM/XGBoost/CatBoost baseline → Optuna 调参 → 模型集成 → 提交。",
        description_en="Standard tabular competition. Full auto pipeline: LightGBM/XGBoost/CatBoost baseline → Optuna tuning → ensemble → submit.",
        example_slugs=["titanic", "playground-series-s6e5", "house-prices"],
    ),

    # ── JSON / Code Competition (research only) ──
    "code_competition": CompetitionType(
        type_id="code_competition",
        name_zh="代码竞赛 / 算法优化",
        name_en="Code Competition / Algorithm",
        detection_extensions=[".json"],
        detection_requires={"min_json_files": 3, "max_csv_files": 0,
                           "has_py_files": False},
        can_baseline=False,
        can_tune=False,
        can_ensemble=False,
        can_submit=True,
        can_research=True,
        description_zh="这个比赛使用 JSON/代码格式，需要编写算法或优化策略。没有通用的 ML 模型可以直接套用。",
        description_en="This competition uses JSON/code format. Requires custom algorithms. No universal ML model applies.",
        example_slugs=["neurogolf-2026", "maze-crawler"],
    ),

    # ── Game / RL (research only, submit OK) ──
    "game_rl": CompetitionType(
        type_id="game_rl",
        name_zh="游戏 / 强化学习",
        name_en="Game / Reinforcement Learning",
        detection_extensions=[".json", ".py"],
        detection_requires={"min_json_files": 3, "has_py_files": True, "has_env_submission": True},
        can_baseline=False,
        can_tune=False,
        can_ensemble=False,
        can_submit=True,
        can_research=True,
        description_zh="这是游戏或强化学习竞赛，需要通过 Python 接口与环境交互。Agent 无法自动生成策略，但可以帮你分析环境和研究方案。",
        description_en="Game/RL competition. Requires Python interaction with environment. Agent can research but not auto-generate agents.",
        example_slugs=["orbit-wars", "maze-crawler"],
    ),

    # ── Image / Computer Vision (not yet implemented, but registered) ──
    "image": CompetitionType(
        type_id="image",
        name_zh="图像 / 计算机视觉",
        name_en="Image / Computer Vision",
        detection_extensions=[".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"],
        detection_requires={"min_img_files": 10},
        can_baseline=False,  # can be set to True when image support is added
        can_tune=False,
        can_ensemble=False,
        can_submit=True,
        can_research=True,
        description_zh="这是图像/计算机视觉竞赛。当前 Agent 不支持自动图像建模，但可以帮你分析数据和研究方案。（图像支持正在开发中）",
        description_en="Image/CV competition. Auto image modeling not yet supported. Agent can research and analyze data. (Image support in development)",
        example_slugs=["birdclef-2026"],
    ),

    # ── Text / NLP (not yet implemented) ──
    "text": CompetitionType(
        type_id="text",
        name_zh="文本 / NLP",
        name_en="Text / NLP",
        detection_extensions=[".txt", ".jsonl"],
        detection_requires={"min_text_files": 10},
        can_baseline=False,
        can_tune=False,
        can_ensemble=False,
        can_submit=True,
        can_research=True,
        description_zh="这是文本/NLP 竞赛。当前 Agent 不支持自动 NLP 建模，但可以帮你分析数据和研究方案。",
        description_en="Text/NLP competition. Auto NLP modeling not yet supported.",
        example_slugs=["llm-science-exam"],
    ),

    # ── Audio (not yet implemented) ──
    "audio": CompetitionType(
        type_id="audio",
        name_zh="音频 / Audio",
        name_en="Audio",
        detection_extensions=[".wav", ".mp3", ".ogg", ".flac"],
        detection_requires={"min_audio_files": 10},
        can_baseline=False,
        can_tune=False,
        can_ensemble=False,
        can_submit=True,
        can_research=True,
        description_zh="这是音频竞赛。当前 Agent 不支持自动音频建模，但可以帮你分析数据和研究方案。",
        description_en="Audio competition. Auto audio modeling not yet supported.",
        example_slugs=["birdclef-2024"],
    ),

    # ── Unknown / Other (fallback) ──
    "unknown": CompetitionType(
        type_id="unknown",
        name_zh="未知类型",
        name_en="Unknown",
        detection_extensions=[],
        can_baseline=False,
        can_tune=False,
        can_ensemble=False,
        can_submit=True,
        can_research=True,
        description_zh="无法自动识别比赛类型。建议手动查看数据后告诉我比赛格式。",
        description_en="Competition type could not be auto-detected. Please check the data manually.",
    ),
}


# ═══════════════════════════════════════════════════════════════════════════════
# Auto-Detection / 自动检测
# ═══════════════════════════════════════════════════════════════════════════════

def detect_competition_type(competition_slug: str) -> CompetitionType:
    """Auto-detect competition type from data files.

    Scans the data directory, counts file types, and matches against
    detection rules in COMPETITION_TYPES. Uses file counts + heuristics
    to decide which type fits best.

    Detection order matters: first match wins. Order: most specific first.
    """
    data_dir = config.COMPETITIONS_DIR / competition_slug / "data" / "raw"
    if not data_dir.exists():
        return COMPETITION_TYPES["unknown"]

    all_files = list(data_dir.rglob("*"))
    files = [f for f in all_files if f.is_file()]

    if not files:
        return COMPETITION_TYPES["unknown"]

    # ── Count by extension ──
    ext_counts: dict[str, int] = {}
    file_names = []
    for f in files:
        ext = f.suffix.lower()
        ext_counts[ext] = ext_counts.get(ext, 0) + 1
        file_names.append(f.name)

    csv_count = ext_counts.get(".csv", 0)
    json_count = ext_counts.get(".json", 0)
    py_count = ext_counts.get(".py", 0)
    img_count = sum(ext_counts.get(e, 0) for e in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"])
    audio_count = sum(ext_counts.get(e, 0) for e in [".wav", ".mp3", ".ogg", ".flac"])
    txt_count = sum(ext_counts.get(e, 0) for e in [".txt", ".jsonl"])

    # ── Heuristic detection ──

    # 1. Game/RL: has Python env code (main.py, env.py, agent.py) + JSON tasks
    has_env_code = any(
        "env" in n.lower() or "gym" in n.lower() or "agent" in n.lower()
        or "submission" in n.lower() or "main" in n.lower()
        for n in file_names
    ) and py_count >= 1
    if json_count >= 10 and has_env_code:
        return COMPETITION_TYPES["game_rl"]

    # 2. Code competition: many JSON files, no Python env files
    if json_count >= 3 and csv_count == 0 and not has_env_code:
        return COMPETITION_TYPES["code_competition"]

    # 3. Tabular: has CSV with enough rows
    if csv_count >= 1:
        # Quick check: read first CSV to get row count
        try:
            import pandas as pd
            for f in files:
                if f.suffix == ".csv" and "train" in f.name.lower():
                    df = pd.read_csv(f, nrows=2)
                    if len(df) >= 1:
                        return COMPETITION_TYPES["tabular"]
            # Any CSV found
            csvs = [f for f in files if f.suffix == ".csv"]
            if csvs:
                return COMPETITION_TYPES["tabular"]
        except Exception:
            if csv_count >= 1:
                return COMPETITION_TYPES["tabular"]

    # 4. Image
    if img_count >= 5:
        return COMPETITION_TYPES["image"]

    # 5. Audio
    if audio_count >= 5:
        return COMPETITION_TYPES["audio"]

    # 6. Text
    if txt_count >= 5:
        return COMPETITION_TYPES["text"]

    # 7. Fallback
    return COMPETITION_TYPES["unknown"]


def get_type_summary(comp_type: CompetitionType) -> str:
    """Generate a Markdown summary of what the agent can/cannot do for this type."""
    can = []
    cannot = []
    if comp_type.can_baseline:
        can.append("✅ 自动生成 ML baseline (LightGBM/XGBoost/CatBoost)")
    else:
        cannot.append("❌ 无法自动生成 ML baseline")
    if comp_type.can_tune:
        can.append("✅ Optuna 超参数调优")
    else:
        cannot.append("❌ 无法自动调参")
    if comp_type.can_ensemble:
        can.append("✅ 模型集成 (simple/weighted/rank average)")
    else:
        cannot.append("❌ 无法自动集成")
    if comp_type.can_research:
        can.append("✅ 公开 Notebook 调研 + 策略文档")
    if comp_type.can_submit:
        can.append("✅ 提交到 Kaggle（人工确认后）")
    can.append("✅ 数据结构分析")
    can.append("✅ 提交文件格式验证")

    lines = [
        f"## {comp_type.name_zh} / {comp_type.name_en}",
        "",
        "### 我能做的 / Capabilities",
    ]
    lines.extend(can)
    lines.append("")
    lines.append("### 我不能做的 / Limitations")
    lines.extend(cannot)
    lines.append("")
    if comp_type.description_zh:
        lines.append(f"> {comp_type.description_zh}")

    return "\n".join(lines)


# ═══════════════════════════════════════════════════════════════════════════════
# Harness Pre-Hook / Harness 前置钩子
# ═══════════════════════════════════════════════════════════════════════════════

class CompetitionGate:
    """Harness pre-hook: blocks tools that don't apply to the current competition type.

    Usage in Harness:
        gate = CompetitionGate()
        harness.add_pre_hook(gate.check)

    The gate tracks the current competition slug → type mapping.
    When a tool is about to execute, it checks: is this tool allowed for this type?
    """

    def __init__(self):
        self._type_cache: dict[str, CompetitionType] = {}  # slug → type

    def set_competition_type(self, slug: str, comp_type: CompetitionType):
        """Explicitly set a competition's type (overrides auto-detection)."""
        self._type_cache[slug] = comp_type

    def get_competition_type(self, slug: str) -> CompetitionType:
        """Get competition type. Auto-detects if not cached."""
        if slug not in self._type_cache:
            self._type_cache[slug] = detect_competition_type(slug)
        return self._type_cache[slug]

    def check(self, tool_name: str, args: dict, risk_level: str):
        """Pre-hook: raise RuntimeError if tool is not allowed for this competition type.

        Args:
            tool_name: The tool being called.
            args: Tool arguments (looks for competition_slug).
            risk_level: Risk classification (not used here, but required by hook interface).

        Raises:
            RuntimeError: If the tool is not allowed for the current competition type.
        """
        slug = args.get("competition_slug", "")
        if not slug:
            return  # Can't check without a slug — allow through

        # Research is always allowed (it sets the type)
        if tool_name in ("research_competition", "inspect_competition"):
            return

        comp_type = self.get_competition_type(slug)

        if not comp_type.can_use(tool_name):
            raise RuntimeError(comp_type.blocked_reason(tool_name))

    def status(self, slug: Optional[str] = None) -> str:
        """Get a status report for all known competitions or a specific one."""
        if slug:
            comp_type = self.get_competition_type(slug)
            return get_type_summary(comp_type)

        lines = ["## 已检测的比赛类型 / Detected Competition Types", ""]
        for s, t in self._type_cache.items():
            can_count = sum(1 for v in t.allowed_tools.values() if v)
            total_tools = len(t.allowed_tools)
            lines.append(f"- **{s}**: {t.name_zh} ({can_count}/{total_tools} tools available)")
        if not self._type_cache:
            lines.append("  (暂无 — 运行 research 后自动填充)")
        return "\n".join(lines)


# ── Singleton ──
_competition_gate: Optional[CompetitionGate] = None


def get_competition_gate() -> CompetitionGate:
    """Get or create the global CompetitionGate singleton."""
    global _competition_gate
    if _competition_gate is None:
        _competition_gate = CompetitionGate()
    return _competition_gate
