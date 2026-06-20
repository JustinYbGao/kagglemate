# KaggleMate / 竞赛导师

> **Your Kaggle mentor — not just a tool that executes, but a teacher that helps you learn and improve.**
> **不只是帮你执行任务的工具，更是帮你**学习**和**成长**的竞赛导师。**

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)](https://langchain.com/langgraph)
[![Multi-LLM](https://img.shields.io/badge/LLM-DeepSeek%20%7C%20OpenAI%20%7C%20Ollama-purple)](https://github.com/JustinYbGao/kagglemate)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)
[![Stars](https://img.shields.io/github/stars/JustinYbGao/kagglemate?style=social)](https://github.com/JustinYbGao/kagglemate)

---

## What is it? / 这是什么？

一个**竞赛导师**。你在 terminal 里跟它对话，它帮你打比赛，更重要的是——**帮你理解每一件事为什么这么做**。

A **competition mentor** that works alongside you. It doesn't just execute — it **explains the WHY** behind every action, turning every competition into a learning experience.

## Scope / 适用范围

| 级别 / Level | 能力 | 说明 |
|---|---|---|
| **Stable** | Tabular baseline workflow | Data profiling → CV plan → baseline script → submission validation → experiment tracking。默认使用 **offline synthetic benchmark**，无需 Kaggle API/数据/LLM。 |
| **Experimental** | Tuning / Ensembling / Kernel / Deep research / Real Kaggle API benchmark | 可用，但尚未被 offline benchmark 证明稳定；需要人工监督或真实数据。 |
| **Research-only** | Image / Text / Audio / Code / RL | Agent 可以调研、讲解、讨论策略，**不会自动生成可运行 baseline**。 |

> 当前 MVP 目标是把 **Tabular CSV** 做成稳定、可复现、可追踪的 baseline 工厂。

## Evaluation / 评估体系

KaggleMate is evaluated as an **AI-assisted Kaggle workflow system**, not an autonomous competition winner.

The benchmark suite measures:

- **End-to-end workflow completion** — can the agent run profiling → CV plan → baseline → validation → experiment tracking without crashing?
- **Valid submission generation** — does the generated submission match the required schema, row count, ID order, and value constraints?
- **CV strategy selection** — is the chosen CV splitter appropriate for the task type (classification/regression/time-series/grouped)?
- **Submission schema validation** — are columns, row order, NaN/inf, probability ranges, and duplicate IDs checked?
- **Feature strategy robustness** — does the validator prevent target leakage, nonexistent features, and constant columns?
- **Experiment reproducibility** — are scripts, configs, OOF predictions, fold scores, and run logs persisted with hashes?

### Running the evaluation

```bash
# Unit tests (no Kaggle data, API, or LLM required)
pytest tests/unit -q

# Integration test (uses offline synthetic fixtures by default)
pytest tests/integration -q

# Synthetic benchmark a single competition (no Kaggle credentials)
python benchmarks/run_benchmark.py --competition titanic --synthetic

# Dry-run all configured competitions (generates scripts/reports, skips training)
python benchmarks/run_benchmark.py --all --synthetic --dry-run

# Full offline benchmark for all configured competitions
python benchmarks/run_benchmark.py --all --synthetic

# Regenerate markdown reports from benchmark results
python benchmarks/update_reports.py
```

### Offline Verification

After installing with `[dev]` (no `[llm]` required):

```bash
pytest tests/unit -q
pytest tests/integration -q
python benchmarks/run_benchmark.py --all --synthetic --dry-run
python benchmarks/update_reports.py
```

To run the full synthetic benchmark (generates and executes training scripts),
you also need the ML models:

```bash
pip install -e ".[ml]"
python benchmarks/run_benchmark.py --competition titanic --synthetic
```

The offline path does not require Kaggle credentials, LLM API keys, or LangChain dependencies unless `use_llm=True`.

To run against real Kaggle data, download the data and use `--data-dir`:

```bash
python benchmarks/run_benchmark.py --competition titanic --data-dir competitions/titanic/data/raw
```

### Benchmark Results

See [`reports/benchmark_summary.md`](reports/benchmark_summary.md) for the latest results.

```bash
python benchmarks/run_benchmark.py --all --synthetic
python benchmarks/update_reports.py
```
### Failure Cases

See [`reports/failure_cases.md`](reports/failure_cases.md) for documented robustness tests.

### 任务执行 / Execution
```
km
你: 帮我研究 titanic，生成 baseline，跑一下
Agent: CV 0.8473，实验 #1 已保存。
      这个分数比随机高很多，说明 LightGBM 在 891 条数据上也能工作——
      但 CV/LB 差距大，可能是因为训练集太小，模型学到了噪音。
      下一步建议: 加正则化或减少叶子数。
```

### 代码讲解 / Code Walkthrough
```
你: 帮我讲解这个高分 Notebook: alexisbcook/titanic-tutorial

Agent: (拉取 Notebook → 解析代码 → 逐段讲解)

       ## 第1部分：数据加载
       这里 pd.read_csv() 直接读取。作者没有用 index_col=0，
       因为 Kaggle 的 test.csv 中 PassengerId 也是从 892 开始的，
       不是索引...

       ## 第2部分：特征工程
       从 Name 中提取 Title (Mr/Mrs/Miss) 是 Titanic 的经典技巧。
       **核心 ML 概念**：特征提取。
       **为什么有效**：Title 跟生存率强相关——Mrs 的存活率远高于 Mr。
       直接在 Name 列上做 one-hot 编码会有上千个特征，
       而提取 Title 只需要 5 个类别。

       ## 要点总结
       1. 特征工程 > 模型选择（新特征比换模型收益更大）
       2. 高基数类别要先聚类再编码（不能直接 one-hot）
       3. 缺失值处理要结合业务逻辑（Age 缺失可能跟 Pclass 相关）
```

### 概念教学 / Concept Teaching
```
你: 什么是 target encoding？在这个比赛里怎么用？

Agent: 用你当前比赛的数据举例——

       Titanic 的 Embarked 列有 3 个值：S、C、Q。
       普通做法是 label encoding → S=0, C=1, Q=2。
       但这个数字没有意义——2 并不比 0"大"。

       Target encoding 的做法：用目标变量的统计值替换类别。
       Embarked='C' 的乘客 Survival rate 是 0.55→用 0.55 替 'C'。
       这样 C=0.55 就直接意味着"55% 存活率"。

       ⚠️ 陷阱：必须用 cross-validation 来算 target encoding，
       否则会数据泄露。比如用 fold 2-5 算出的值编码 fold 1。

       要不要我生成一个实验验证一下 target encoding 的效果？
```

### 实验对比 / Experiment Comparison
```
你: 对比实验 2 和实验 3，分析为什么差这么多

Agent: | | 实验 2 (baseline) | 实验 3 (tuned) |
       | CV | 0.8417 | 0.8539 (+0.012) |
       | 学习率 | 0.1 | 0.007 |
       | 正则化 | 无 | reg_alpha=1.53 |

       实验 3 更好的原因：小数据集上正则化比学习率更重要。
       你可以学到的经验：891 条数据用 default params 容易过拟合。
       下次可以先从 regularization 入手而不是调 n_estimators。
```

---

## Grounded Tutoring

KaggleMate includes a grounded tutoring layer to reduce unsupported ML recommendations. Instead of answering only from the model's prior knowledge, tutoring responses can retrieve evidence from:

- CV plans (`CV_PLAN.md`, `cv_config.json`)
- Data profiles (`data_profile.md`, `data_profile.json`)
- Benchmark results (`benchmark_result.json`)
- Experiment records (`experiments.db`)
- Strategy validation reports (`strategy_validation_report.json`)
- Submission validation reports (`submission_validation_report.json`)
- Experiment configs (`experiment_config.json`)
- Run logs (`run_log.txt`)
- Notebook / code chunks (`.py`, `.ipynb`)
- ML concept notes (`docs/ml_concepts/*.md`)

Tutoring responses separate:

- **Confirmed facts** from artifacts
- **Interpretation** of those facts
- **Uncertainty** when evidence is insufficient
- **Next verifiable experiment** to reduce uncertainty

### Experiment Diagnosis

The grounded tutor can read experiment artifacts and, when available, `experiments.db` records to explain model iteration results. It separates observed facts from hypotheses and avoids attributing score changes to features unless ablation evidence exists.

```python
from pathlib import Path
from kagglemate.tutor.grounded_tutor import answer_tutoring_question

result = answer_tutoring_question(
    question="Why did experiment 2 perform worse than experiment 1?",
    project_root=Path("."),
    competition_slug="titanic",
    mode="experiment_diagnosis",
    use_llm=False,
)
print(result["answer"])
```

When `mode="experiment_diagnosis"`, the answer includes four sections:

1. **Experiment facts from artifacts** — what was actually recorded.
2. **Diagnosis** — conservative interpretation of the facts.
3. **What cannot be concluded yet** — missing evidence (LB scores, fold scores, OOF, validation reports).
4. **Next verifiable experiments** — concrete ablations or re-runs to reduce uncertainty.

See [`examples/grounded_tutoring_demo.md`](examples/grounded_tutoring_demo.md) for example tutoring outputs.

### Example

```python
from pathlib import Path
from kagglemate.tutor.grounded_tutor import answer_tutoring_question

result = answer_tutoring_question(
    question="Why does target encoding risk leakage in this competition?",
    project_root=Path("."),
    competition_slug="titanic",
    mode="concept_tutor",
    use_llm=False,
)
print(result["answer"])
```

`use_llm=False` runs completely offline and is the default, which makes it safe for tests and environments without API keys. `use_llm=True` synthesizes a response with the configured LLM provider while still requiring citations to retrieved chunks.

> **Disclaimer**: The grounded tutor does not guarantee model correctness. It reduces hallucination by forcing answers to cite local project artifacts and by explicitly marking uncertain claims that require experiments. Grounded tutoring reduces unsupported recommendations but does not prove that a modeling idea improves leaderboard performance. Performance claims require experiment artifacts or benchmark results.

---

## Installation

For offline benchmark, validation, and grounded tutoring:

```bash
python -m pip install -e ".[dev]"
```

For LLM-powered mentoring / agent features:

```bash
python -m pip install -e ".[dev,llm]"
```

The `[dev]` extra includes pytest and pytest-asyncio; the `[llm]` extra includes LangGraph, LangChain, and the OpenAI SDK. The offline path does not require Kaggle credentials, LLM API keys, or LangChain dependencies unless `use_llm=True`.

---

## Quick Start / 快速开始

### Step 1: Clone & Install / 克隆并安装

```bash
git clone https://github.com/JustinYbGao/kagglemate.git
cd kagglemate
python3.10 -m venv .venv          # Python 3.10+ required / 需要 Python 3.10+
source .venv/bin/activate

# Offline evaluation + grounded tutoring (no LLM/Kaggle API needed)
pip install -e ".[dev]"

# OR full agent features (requires LLM API key in .env)
pip install -e ".[dev,llm]"
```

If you are only running tabular baselines, you also need the ML models:

```bash
pip install -e ".[ml]"
```

### Step 2: Get your API keys / 获取 API 密钥

You need **two** things set up. If either is missing, the health check will tell you. / 你需要配置**两项**。如果缺失，健康检查会提示。

**① LLM API Key**（必需——Agent 的大脑。支持任何 OpenAI 兼容接口）

KaggleMate supports any OpenAI-compatible API. Pick your provider:

```bash
cp .env.example .env
# 编辑 .env:

# DeepSeek (默认)
LLM_PROVIDER=deepseek
LLM_API_KEY=sk-your-deepseek-key     # https://platform.deepseek.com/api_keys

# 或 OpenAI
LLM_PROVIDER=openai
LLM_API_KEY=sk-your-openai-key       # https://platform.openai.com/api-keys

# 或本地 Ollama / vLLM
LLM_PROVIDER=custom
LLM_BASE_URL=http://localhost:11434/v1
LLM_MODEL=qwen3:latest
LLM_API_KEY=ollama
```

| Provider / 提供商 | base_url | 默认模型 | 价格 |
|---|---|---|---|
| DeepSeek | `api.deepseek.com` | `deepseek-v4-pro` | ~$0.44/百万token |
| OpenAI | `api.openai.com/v1` | `gpt-4.1` | ~$5/百万token |
| Ollama (本地) | `localhost:11434/v1` | 你指定的 | 免费 |
| 自定义 | 任意 | 你指定的 | 取决于提供商 |

> ⚠️ 兼容旧版：如果你之前已经设置了 `DEEPSEEK_API_KEY`，无需修改——会自动检测为 DeepSeek 模式。

**② Kaggle API 凭证**（必需——下载数据、提交比赛）

1. 打开 [kaggle.com/settings/account](https://www.kaggle.com/settings/account)
2. 滚动到 "API" 部分 → 点击 "Create New Token" → 下载 `kaggle.json`
3. 将 `kaggle.json` 放到 `~/.kaggle/` 目录：
   ```bash
   mkdir -p ~/.kaggle
   mv ~/Downloads/kaggle.json ~/.kaggle/
   chmod 600 ~/.kaggle/kaggle.json    # 防止其他用户读取 / protect your key
   ```
4. 或者在 `.env` 中直接设置（不推荐，但可用）：
   ```bash
   KAGGLE_USERNAME=your-kaggle-username
   KAGGLE_KEY=your-kaggle-api-key
   ```

> ⚠️ GitHub 不会包含你的 API 密钥。`.env` 和 `~/.kaggle/kaggle.json` 都不会被提交。

### Step 3: Verify / 验证安装

```bash
python main.py check
```

你应该看到全绿 / You should see all green ✓：

```
KaggleMate Health Check
  Python 3.12.13  ✓
  DeepSeek API key  ✓
  Kaggle CLI  ✓
  Kaggle credentials  ✓
  Active competitions: 20+  ✓
```

### Step 4: Start the agent / 启动对话

```bash
python main.py
```

或者设置快捷别名（可选）：

```bash
# 加到 ~/.zshrc
echo 'km() { cd /path/to/kagglemate && source .venv/bin/activate && python main.py chat "$@"; }' >> ~/.zshrc
source ~/.zshrc
km
```

---

## Architecture / 架构

```
┌──────────────────────────────────────────────────────────────────┐
│                      You / 你 (自然语言)                          │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                    LLM (大脑 / Brain)                              │
│     DeepSeek / OpenAI / Ollama — 理解意图 → 选择工具 → 教学回复     │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│              🧑‍🏫 Mentor Layer / 导师层 (NEW)                      │
│   explain_notebook  •  explain_concept  •  compare_approaches    │
│   "不只是执行——解释每一步为什么这样做"                               │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                   ╔═══════ Harness ═══════╗                       │
│                   ║ 1. Competition Gate  ║ ← 类型能力门控          │
│                   ║ 2. Risk Confirmation ║ ← 危险操作人工确认       │
│                   ║ 3. Session Budget    ║ ← 防 runaway            │
│                   ║ 4. Audit Trail       ║ ← 不可变审计日志         │
│                   ╚══════════════════════╝                       │
│                   LLM CANNOT bypass / 模型无法绕过                 │
└──────────────────────────┬───────────────────────────────────────┘
                           │
┌──────────────────────────▼───────────────────────────────────────┐
│                     LangGraph StateGraph                         │
│   Init → Analyze → Research → Plan → Baseline → Run → Evaluate  │
│     │              │            │         │          │    │       │
│     ▼              ▼            ▼         ▼          ▼    ▼       │
│  DeepResearch   Kernel       Suggest    Tune    Ensemble  Submit │
│  (Kaggle+arXiv  (pull/push/  (策略建议)  (Optuna) (3种方法) (Human│
│   +Web合成)     monitor)                                  Gate)  │
└──────────────────────────────────────────────────────────────────┘
```

---

## Benchmark Results / 基准结果

Offline synthetic benchmark 结果（无需 Kaggle API/数据/LLM）：

| Competition | Type | Metric | CV | Runtime | Status |
|---|---|---|---|---|---|
| titanic | binary classification | accuracy | see latest run | see latest run | baseline |
| house-prices-advanced-regression-techniques | regression | rmse_log | see latest run | see latest run | baseline |
| spaceship-titanic | binary classification | accuracy | see latest run | see latest run | baseline |

运行 `python benchmarks/run_benchmark.py --all --synthetic` + `python benchmarks/update_reports.py` 查看最新结果。

See [`reports/benchmark_summary.md`](reports/benchmark_summary.md) for the latest results.

> 注：Public LB 分数需要真实 Kaggle 数据与提交，属于 Experimental 能力，不在默认 offline benchmark 范围内。

---

## Capabilities / 能力矩阵

### 按比赛类型 / By Competition Type

| 类型 / Type | 示例 | 调研 | 深度调研 | Baseline | 调参 | 集成 | 提交 |
|---|---|---|---|---|---|---|---|
| **Tabular CSV** | titanic, playground-s6e5 | ✅ | ✅ | ✅ LightGBM/XGBoost/CatBoost | ✅ Optuna | ✅ 3 methods | ✅ |
| 代码竞赛 | neurogolf-2026 | ✅ | ✅ | — research only | — | — | ✅ |
| 游戏/RL | orbit-wars | ✅ | ✅ | — research only | — | — | ✅ |
| 图像/文本/音频 | birdclef, NLP | ✅ | ✅ | — research only | — | — | ✅ |

### 21 Conversational Tools / 21 个对话工具

| 分类 | 工具 | 说明 |
|---|---|---|
| **🧑‍🏫 学习 / Mentor** | **`explain_notebook`** | **拉取+逐段讲解任何 Kaggle Notebook，解释每部分的 ML 概念和思路** |
| | **`explain_concept`** | **用当前比赛数据讲解任何 ML 概念（target encoding、SHAP、CV 策略...）** |
| | **`compare_approaches`** | **对比两个实验，分析分数差异原因，总结可迁移的经验** |
| 调研 | `research_competition` | 基础调研：数据画像 + Kaggle Notebook + SPEC.md |
| | **`deep_research`** | **深度调研：Kaggle + arXiv 论文 + Web 搜索 + 交叉分析** |
| | `what_can_i_do` | 查询当前比赛 Agent 能做什么 |
| | `pull_notebook` | 拉取公开 Notebook（保留 metadata） |
| 建模 | `generate_baseline` | LLM 设计特征 + 模板生成 LightGBM/XGBoost/CatBoost 脚本 |
| | `run_experiment` | 执行训练，解析 CV，入库 |
| | `tune_model` | Optuna 超参数调优 |
| | `ensemble_blend` | 简单平均 / 加权平均 / 排序平均 |
| 策略 | `get_suggestions` | 基于实验历史推荐下一步 |
| 实验 | `list_experiments` / `show_experiment` | 查看实验列表和详情 |
| | `record_lb_score` | 记录排行榜分数 |
| 提交 | `validate_submission` | 9 项格式检查 |
| | `submit_to_kaggle` / `check_submission_status` | 提交（⚠️ 硬拦截：必须人工确认） + 查看榜单 |
| 比赛 | `list_competitions` / `inspect_competition` / `read_generated_file` | 浏览比赛 / 查看文件 / 读取报告 |

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
├── chat_agent.py              # 对话 Agent (21 tools + 自然语言调度)
├── mentor.py                  # 🧑‍🏫 导师层: Notebook讲解/概念教学/实验对比
├── harness.py                 # 安全护栏 (5 层安全: 预算/类型/风险/审计/钩子)
├── competition_registry.py    # 比赛类型注册表 (7 种, 自动检测+能力门控)
├── config.py                  # 多 provider 配置 (DeepSeek/OpenAI/Ollama/...)
├── graph/                     # LangGraph 状态机
│   ├── builder.py             # 16 节点 + 条件边
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
- [x] Phase 6f: Mentor Mode (notebook walkthrough, concept teaching, experiment comparison) / 导师模式
- [x] Phase 6g: Multi-Provider LLM (DeepSeek / OpenAI / Ollama / custom) / 多模型支持
- [ ] Image/Text baseline support / 图像/文本 baseline — **research-only (no AutoML baseline planned for MVP)**

---

## Prerequisites / 环境要求

| Requirement / 需求 | 说明 |
|---|---|
| **Python 3.10+** | [python.org](https://python.org) 或 `brew install python@3.12` |
| **DeepSeek API Key** | [platform.deepseek.com/api_keys](https://platform.deepseek.com/api_keys) — 免费注册，按量付费（V4 Pro $0.44/百万token） |
| **Kaggle API 凭证** | [kaggle.com/settings/account](https://www.kaggle.com/settings/account) → Create New Token → 放到 `~/.kaggle/kaggle.json` |
| **已接受的比赛规则** | 如果你要下载某个比赛的数据，需要先在网页上点击 "Join Competition" 接受规则 |

### Optional Dependencies / 可选依赖

```bash
pip install -e ".[ml]"        # LightGBM + XGBoost + CatBoost（Tabular baseline 必需）
pip install -e ".[tune]"      # Optuna 超参数优化
pip install -e ".[llm]"       # LangGraph + LangChain + OpenAI SDK（Agent / LLM 功能必需）
pip install -e ".[dev]"       # pytest 测试框架
pip install -e ".[dev,llm]"   # 完整开发 + LLM 功能
```

## Troubleshooting / 常见问题

| Problem / 问题 | Cause / 原因 | Solution / 解决 |
|---|---|---|
| `ModuleNotFoundError: kaggle` | Kaggle CLI 未安装 | `pip install kaggle` |
| `kaggle.json not found` | 未配置 Kaggle API | 去 [kaggle.com/settings/account](https://www.kaggle.com/settings/account) → Create New Token |
| `403 Forbidden` when downloading | 未接受比赛规则 | 在 Kaggle 网页上点击该比赛的 "Join Competition" |
| `DEEPSEEK_API_KEY not set` | `.env` 未配置 | `cp .env.example .env`，填入真实 key |
| arXiv 搜索无结果 / timeout | 国内网络限制 arXiv API | 正常现象——Deep Research 自动跳过，Kaggle+Web 结果仍可用 |
| `name 'Panel' is not defined` | Rich 库版本问题 | `pip install --upgrade rich` |
| Baseline 脚本语法错误 | LLM 生成代码偶有瑕疵 | 重新运行 `python main.py baseline`，或手动微调脚本 |
| Baseline 验证失败 / FE 代码异常 | strategy_validator 拦截了不合法策略 | 查看 `scripts/experiment_config.json` 了解最终使用的策略；系统已自动 fallback 到 heuristic |

---

## License / 许可证

MIT
