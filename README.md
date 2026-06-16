# KaggleMate / 卡钩助手

> **Conversational Kaggle competition agent — LangGraph + DeepSeek V4 + multi-source research**
> **对话式 Kaggle 竞赛 Agent——用自然语言打比赛**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)](https://langchain.com/langgraph)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V4%20Pro-purple)](https://platform.deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Stars](https://img.shields.io/github/stars/JustinYbGao/kagglemate?style=social)](https://github.com/JustinYbGao/kagglemate)

---

## What is it? / 这是什么？

一个**对话式 Kaggle 竞赛助手**。用自然语言告诉它你想做什么，它会自己调工具、跑代码、分析结果、给你建议。

A **conversational Kaggle competition agent**. Talk to it in natural language — it calls tools, runs code, analyzes results, and recommends next steps autonomously.

```
km
你: 帮我看看我参加了哪些比赛
Agent: 你正在参加 4 个比赛: orbit-wars、neurogolf-2026、rogii-wellbore...

你: 深度调研 orbit-wars
Agent: (搜索 Kaggle + arXiv + Web) → 交叉分析 → 生成 deep_research.md (方法矩阵+论文迁移建议)

你: 帮我看一下 titanic，生成 baseline 跑一下
Agent: CV 0.8473，实验 #1 已保存。建议: 加正则化、做特征工程...
```

---

## Quick Start / 快速开始

```bash
# 1. Clone & install / 克隆并安装
git clone https://github.com/JustinYbGao/kagglemate.git
cd kagglemate
pip install -e ".[ml]"

# 2. Configure / 配置
cp .env.example .env
# 编辑 .env，填入 DEEPSEEK_API_KEY

# 3. Start the agent / 启动对话
python main.py
# 或: source ~/.zshrc && km
```

---

## Architecture / 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      You / 你 (自然语言)                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    DeepSeek V4 Pro (大脑)                         │
│          理解意图 → 选择工具 → 综合回复                             │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                   ╔═══════ Harness ═══════╗                       │
│                   ║ 1. Competition Gate  ║ ← 类型能力门控          │
│                   ║ 2. Risk Confirmation ║ ← 危险操作人工确认       │
│                   ║ 3. Session Budget    ║ ← 防止 runaway          │
│                   ║ 4. Audit Trail       ║ ← 不可变审计日志         │
│                   ╚══════════════════════╝                       │
│                   LLM CANNOT bypass / 模型无法绕过                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                     LangGraph StateGraph                         │
│                                                                   │
│  Init → Analyze → Research → Plan → Baseline → Run → Evaluate    │
│    │                                       ↓                      │
│    │                              ┌───────┴────────┐              │
│    │                              ▼                ▼              │
│    │                           Suggest ──▶       END              │
│    │                              │          (auto-stop)          │
│    │                              ▼                                │
│    └── DeepResearch (并行搜索)     END                              │
│    └── Kernel (pull/push/monitor)                                  │
│    └── Tune (Optuna)                                               │
│    └── Ensemble (加权/排序平均)                                     │
│    └── Submit (Human Gate)                                         │
└──────────────────────────────────────────────────────────────────┘
```

---

## Capabilities / 能力矩阵

### 按比赛类型 / By Competition Type

| 类型 / Type | 示例 | 调研 | 深度调研 | Baseline | 调参 | 集成 | 提交 |
|---|---|---|---|---|---|---|---|
| Tabular CSV | titanic, playground-s6e5 | ✅ | ✅ | ✅ LightGBM/XGBoost/CatBoost | ✅ Optuna | ✅ 3 methods | ✅ |
| 代码竞赛 | neurogolf-2026 | ✅ | ✅ | — 需手写算法 | — | — | ✅ |
| 游戏/RL | orbit-wars | ✅ | ✅ | — 需手写 Agent | — | — | ✅ |
| 图像/文本/音频 | birdclef, NLP | ✅ | ✅ | — 开发中 | — | — | ✅ |

### 18 Conversational Tools / 18 个对话工具

| 分类 | 工具 | 说明 |
|---|---|---|
| 比赛发现 | `list_competitions` | 浏览活跃比赛 / 查看已参加的比赛 |
| | `inspect_competition` | 查看比赛文件（不下载） |
| 调研 | `research_competition` | 基础调研：数据画像 + Kaggle Notebook + SPEC.md |
| | **`deep_research`** | **深度调研：Kaggle + arXiv 论文 + Web 搜索 + 交叉分析** |
| | `what_can_i_do` | 查询当前比赛 Agent 能做什么 |
| | `pull_notebook` | 拉取公开 Notebook（保留 metadata） |
| 建模 | `generate_baseline` | LLM 设计特征 + 模板生成 LightGBM 脚本 |
| | `run_experiment` | 执行训练，解析 CV，入库 |
| | `tune_model` | Optuna 超参数调优 |
| | `ensemble_blend` | 简单平均 / 加权平均 / 排序平均 |
| 策略 | `get_suggestions` | 基于实验历史推荐下一步 |
| 实验 | `list_experiments` | 查看实验列表 |
| | `show_experiment` | 查看实验详情 |
| | `record_lb_score` | 记录排行榜分数 |
| 提交 | `validate_submission` | 9 项格式检查 |
| | `submit_to_kaggle` | 提交（⚠️ 硬拦截：必须人工确认） |
| | `check_submission_status` | 查看提交状态和榜单 |
| 文件 | `read_generated_file` | 读取生成的报告 |

---

## Safety / 安全护栏 (Harness)

System Prompt 是**建议**，Harness 是**硬拦截**。模型无法绕过。

System Prompts tell the model what to do. The Harness **enforces it in code** — the LLM cannot bypass these gates.

| 层 / Layer | 做了什么 / What it does |
|---|---|
| **Competition Gate / 类型门控** | 自动检测比赛类型，拦截不适用的工具（对 neurogolf 调用 baseline → 被拦截并解释原因） |
| **Risk Confirmation / 风险确认** | DANGEROUS 级操作（提交）→ 必须人工输入 YES |
| **Session Budget / 会话预算** | 50 次工具调用上限，3 次危险操作上限，防 runaway |
| **Audit Trail / 审计日志** | 每次操作写入 `~/.kagglemate/audit.log`，不可变 JSON-line 格式 |
| **Pre/Post Hooks / 钩子系统** | 可扩展的安全规则插件 |

### 比赛类型自动检测 / Auto-Detection

7 种比赛类型：下载数据后自动扫描文件扩展名，判断类型，注册到 Harness。不同类型的工具权限不同。

| 类型 | 检测条件 | 自动 ML |
|---|---|---|
| Tabular CSV | 有 train.csv | ✅ |
| 代码竞赛 | JSON ≥ 3, 无 CSV | ❌ |
| 游戏/RL | JSON + Python env 代码 | ❌ |
| 图像 | png/jpg ≥ 10 | 开发中 |
| 文本/NLP | txt/jsonl ≥ 10 | 开发中 |
| 音频 | wav/mp3 ≥ 10 | 开发中 |
| 未知 | 兜底 | ❌ |

> 要添加新类型：编辑 `kagglemate/competition_registry.py` 加一个条目。Harness 自动生效。

---

## Deep Research / 深度调研

不仅搜索 Kaggle Notebooks，还搜索**学术论文 + 全网资源**，交叉分析。

Searches **Kaggle + arXiv + the entire web**, cross-analyzes with LLM synthesis.

```
你: 深度调研 playground-series-s5e6

Agent 内部:
  Phase 1: Kaggle notebooks → 20 results
  Phase 2: LLM 提取技术关键词 (5个)
  Phase 3: 并行搜索 (4 workers)
    → arXiv: gradient boosting ensemble → 5 papers
    → Web: time series forecasting → 5 results
    → Web: feature engineering tabular → 5 results
    → ...
  Phase 4: LLM 交叉合成 → deep_research.md
    ├── Method Matrix / 方法矩阵
    ├── Consensus Techniques / 共识方法
    ├── Novel Approaches / 前沿方法
    ├── Paper-to-Practice Bridge / 论文落地建议
    ├── Recommended Strategy / 推荐策略
    └── Key References / 关键引用

输出: 13.2 KB 结构化报告
```

**可插拔架构**：新增数据源只需实现 `ResearchProvider` 基类。

---

## Chat Commands / 对话命令

在对话中可用：

```
/harness, /status  查看安全护栏状态
/audit             查看审计日志 (最近 20 条)
/types, /competition  查看比赛类型注册表
/yesall            当前会话批量确认危险操作
/noyesall          恢复逐个确认
```

---

## CLI Commands / 命令行

```bash
# 对话 Agent（默认）
km                                    # 启动对话

# 调研
km research <slug>                    # 基础调研
km deep-research <slug>               # 深度调研（多源）

# 建模
km baseline <slug>                    # 生成训练脚本
km run <slug>                         # 执行实验
km tune <slug> --trials 50            # Optuna 调参
km ensemble <slug> --ids 1,2,3        # 模型融合
km suggest <slug>                     # 策略建议

# 实验
km experiments <slug> -a list         # 列表
km experiments <slug> -a show --id 1   # 详情
km experiments <slug> -a log-lb --id 1 --lb 0.85  # 记录 LB

# Kernel
km notebook pull <ref> -c <slug>
km kernel push <dir> -c <slug>
km kernel monitor <ref>

# 提交 (Human Gate)
km submission validate -c <slug> -f <file>
km submission submit -c <slug> -f <file>

# 安全
km harness status                     # 护栏状态
km harness audit                      # 审计日志
km types                              # 比赛类型注册表
```

---

## Project Structure / 项目结构

```
kagglemate/
├── chat_agent.py              # 对话 Agent (18 tools + DeepSeek 调度)
├── harness.py                 # 安全护栏 (5 层安全: 预算/类型/风险/审计/钩子)
├── competition_registry.py    # 比赛类型注册表 (7 types, 自动检测+能力矩阵)
├── config.py                  # 配置管理
├── graph/                     # LangGraph 状态机
│   ├── builder.py             # 15 节点 + 条件边
│   ├── state.py               # KaggleAgentState TypedDict
│   └── nodes/
│       ├── init_node.py       # 下载数据
│       ├── analyze_node.py    # 数据画像
│       ├── research_node.py   # Kaggle 调研
│       ├── deep_research_node.py  # 多源深度调研 (Kaggle+arXiv+Web)
│       ├── plan_node.py       # SPEC.md 生成
│       ├── baseline_node.py   # 训练脚本生成
│       ├── run_node.py        # 执行实验
│       ├── evaluate_node.py   # 评估诊断
│       ├── suggest_node.py    # 策略建议
│       ├── tune_node.py       # Optuna 调参
│       ├── ensemble_node.py   # 模型融合
│       ├── kernel_node.py     # Kaggle kernel 管理
│       └── submit_node.py     # 提交验证
├── research_providers/        # 调研数据源 (可插拔)
│   ├── base.py                # ResearchProvider 抽象基类
│   ├── kaggle_provider.py     # Kaggle Notebooks
│   ├── arxiv_provider.py      # arXiv 论文 (免费 API)
│   └── web_provider.py        # DuckDuckGo 搜索 (免费)
├── tools/                     # 领域工具
│   ├── kaggle_cli.py          # Kaggle CLI 封装
│   ├── data_profiler.py       # CSV 数据分析
│   ├── submission_validator.py  # 提交验证 (9 项检查)
│   └── llm_client.py          # DeepSeek API 适配
├── memory/                    # 实验追踪
│   ├── schema.sql             # SQLite 建表
│   └── experiment_store.py    # CRUD
├── templates/                 # Jinja2 模板 (7 个)
│   ├── spec_template.md       # SPEC 模板 (双语)
│   ├── research_summary_template.md
│   ├── rules_checklist_template.md
│   ├── next_steps_template.md
│   ├── baseline_script_template.py.j2
│   ├── tune_script_template.py.j2
│   └── deep_research_template.md
├── tests/
│   └── test_deepseek_tools.py # DeepSeek 工具调用验证
└── main.py                    # CLI 入口 (Typer)
```

---

## Roadmap / 路线图

- [x] Phase 0: Project skeleton + DeepSeek validation / 项目骨架 + DeepSeek 验证
- [x] Phase 1: Research Agent (data + notebooks + SPEC) / 调研 Agent
- [x] Phase 2: Baseline Agent (script + execution + experiment DB) / Baseline Agent
- [x] Phase 3: Evaluate + Suggest (strategy advisor) / 评估+建议
- [x] Phase 4: Kaggle Kernel Agent (pull/push/monitor) / Kernel Agent
- [x] Phase 5: Semi-Auto Submit (validator + Human Gate) / 半自动提交
- [x] Phase 6a: AutoML (Optuna tuning + ensemble) / 自动调参+集成
- [x] Phase 6b: Conversational Agent (18 tools, natural language) / 对话 Agent
- [x] Phase 6c: Agent Harness (safety gates + audit + budget) / 安全护栏
- [x] Phase 6d: Competition Registry (7 types, auto-detect, capability gate) / 比赛类型注册表
- [x] Phase 6e: Deep Research (Kaggle + arXiv + Web synthesis) / 深度调研
- [ ] Image/Text baseline support / 图像/文本 baseline

---

## License / 许可证

MIT
