# KaggleMate / 卡钩助手

> **LangGraph-based Kaggle competition assistant agent powered by DeepSeek V4**
> **基于 LangGraph + DeepSeek V4 的 Kaggle 竞赛辅助 Agent**

给定一个 Kaggle 比赛 slug，自动下载数据、调研公开 Notebook、生成策略文档、编写并运行 baseline 模型、追踪实验记录、并给出下一步优化建议——提交前始终保留人工确认环节。

Takes a Kaggle competition slug and autonomously researches public notebooks, profiles data, generates strategy documents, writes and runs baseline ML models, tracks experiments, and recommends next steps — all with a Human Gate for submissions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)](https://langchain.com/langgraph)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V4%20Pro-purple)](https://platform.deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Quick Start / 快速开始

```bash
# 1. Clone & install / 克隆并安装
git clone https://github.com/JustinYbGao/kagglemate.git
cd kagglemate
pip install -e ".[ml]"

# 2. Configure / 配置环境变量
cp .env.example .env
# 编辑 .env，填入你的 DEEPSEEK_API_KEY

# 3. Verify / 验证安装
python main.py check

# 4. Run research + baseline on any competition / 对任意比赛运行调研+Baseline
python main.py research titanic
python main.py baseline titanic
python main.py run titanic
```

## Architecture / 架构

```
LangGraph StateGraph (11 nodes / 节点):
  
  Init ──▶ Analyze ──▶ Research ──▶ Plan ──▶ Baseline ──▶ Run ──▶ Evaluate
                                                                      │
                                                              ┌───────┴───────┐
                                                              ▼               ▼
                                                           Suggest ──▶      END
                                                              │         (stop / 停止)
                                                              ▼
                                                            END
```

### Nodes / 节点说明

| Node / 节点 | Phase / 阶段 | What it does / 功能 |
|------|-------|-------------|
| `init` | 1 | Downloads data, creates workspace / 下载数据、创建目录 |
| `analyze` | 1 | Profiles CSV data, identifies task type & metric / 分析CSV、识别任务类型和指标 |
| `research` | 1 | Fetches top public notebooks, LLM summarizes techniques / 获取高分Notebook、LLM总结技术方案 |
| `plan` | 1 | Generates SPEC.md + research_summary.md + rules_checklist.md / 生成策略文档 |
| `baseline` | 2 | LLM designs features + Jinja2 renders training script / LLM设计特征+模板渲染训练脚本 |
| `run` | 2 | Executes script, parses CV score, saves to experiments.db / 执行脚本、解析CV、入库 |
| `evaluate` | 3 | CV/LB gap analysis, overfitting detection, plateau checks / CV/LB差距分析、过拟合检测 |
| `suggest` | 3 | LLM recommends next 3-5 experiments with expected improvement / LLM推荐下一步实验 |
| `kernel` | 4 | Pull/push/monitor Kaggle kernels with metadata validation / Kaggle kernel 管理 |
| `submit` | 5 | Submission validation + Human Gate confirmation / 提交验证 + 人工确认 |

### Conditional Edge / 条件边

```
Evaluate ──▶ _should_continue=True ──▶ Suggest ──▶ END
          ──▶ _should_continue=False ──▶ END (≥3 consecutive failures / 连续 3 次失败则停止)
```

## CLI Commands / CLI 命令

```bash
# ── Research / 调研 ──
python main.py research <slug>        # Full pipeline: data + notebooks + SPEC / 完整调研流水线
python main.py profile <slug>         # Data profiling only / 仅数据画像
python main.py spec <slug>            # Generate SPEC.md from existing data / 从已有数据生成SPEC

# ── Modeling / 建模 ──
python main.py baseline <slug>        # Generate training script / 生成训练脚本
python main.py run <slug>             # Execute + record experiment / 执行并记录实验
python main.py suggest <slug>         # Get next-step recommendations / 获取下一步建议

# ── Experiments / 实验管理 ──
python main.py experiments <slug> --action list              # 列表
python main.py experiments <slug> --action show --id <id>    # 详情
python main.py experiments <slug> --action compare --ids 1,2,3  # 对比
python main.py experiments <slug> --action log-lb --id <id> --lb <score>  # 记录LB

# ── Kaggle Kernels / 云端 Notebook ──
python main.py notebook pull <ref> -c <slug>     # 拉取公开 Notebook（保留 metadata）
python main.py kernel push <dir> -c <slug>       # 推送 Kernel
python main.py kernel monitor <ref>              # 监控 Kernel 状态

# ── Submissions / 提交 (always requires human "YES" / 始终需人工确认) ──
python main.py submission validate -c <slug> -f <file>.csv   # 验证格式
python main.py submission submit -c <slug> -f <file>.csv -m "message"  # 提交
python main.py submission status -c <slug>                    # 查看提交记录
```

## Example: Titanic / 示例：泰坦尼克号

```bash
$ python main.py research titanic
  [init] → init
  [analyze] → analyze
  [research] → research  
  [plan] → plan
✅ Research complete! / 调研完成！
  📄 SPEC.md (8.0 KB)
  📄 data_profile.md (1.2 KB)
  📄 rules_checklist.md (2.2 KB)
  Task type / 任务类型: tabular_classification
  Metric / 指标: accuracy

$ python main.py baseline titanic
  Model / 模型: LightGBM
  Features / 特征: 11 selected (LLM-chosen feature engineering / LLM 自动选择)

$ python main.py run titanic
  CV: 0.8473 ± 0.0149 (ACCURACY)
  Top features / 重要特征: Fare, Age, Pclass, FamilySize, Title...
  ✅ Experiment #1 saved / 实验 #1 已保存

$ python main.py suggest titanic
  💡 5 recommendations generated / 5 条建议:
  1. LightGBM regularization / 正则化 (+0.010)
  2. Feature: Title + FamilySize / 特征工程 (+0.015)
  3. StratifiedKFold fix / 分层验证修正 (+0.000)
  4. Ensemble LightGBM + LogisticRegression / 模型集成 (+0.005)
  5. Fare/Age binning / 票价/年龄分箱 (+0.005)
```

## Key Design Decisions / 核心设计决策

| Decision / 决策 | Choice / 选择 | Why / 原因 |
|----------|--------|-----|
| Agent framework / Agent 框架 | LangGraph Graph API | Explicit state machine, conditional edges, visualization / 显式状态机、条件边、可可视化 |
| LLM provider / LLM 提供商 | DeepSeek V4 Pro | 1M context, strong tool calling, affordable / 百万上下文、工具调用能力强、性价比高 |
| Code generation / 代码生成 | Template + LLM fill / 模板+LLM填空 | Training scripts are Jinja2 templates; LLM only writes feature engineering / 脚本骨架用模板、LLM只写特征工程 |
| Submission / 提交流程 | CLI Human Gate / 命令行人工确认 | Never auto-submit. User must type YES after validation. / 绝不自动提交，验证通过后必须输入 YES |
| Experiment tracking / 实验追踪 | SQLite per competition / 每比赛一个 SQLite | Portable, zero-config, easy to backup / 便携、零配置、易于备份 |
| Data format support / 数据格式 | Tabular CSV first / 优先 Tabular CSV | Most reliable format for automation / 最可靠的自动化数据格式 |

## Project Structure / 项目结构

```
kagglemate/
├── graph/           # LangGraph StateGraph nodes & builder / 图节点与构建器
│   ├── builder.py   # Graph assembly + conditional edges / 图组装+条件边
│   ├── state.py     # KaggleAgentState TypedDict / 状态类型定义
│   └── nodes/       # 10 node implementations / 10个节点实现
├── tools/           # Domain tools (wrappers around Kaggle CLI, pandas) / 领域工具
├── memory/          # Experiment tracking (SQLite) / 实验追踪
├── templates/       # Jinja2 templates (SPEC, baseline scripts, reports) / 模板
├── main.py          # Typer CLI entry point / CLI 入口
└── tests/           # DeepSeek tool calling verification / 工具调用验证
```

## Prerequisites / 环境要求

- Python 3.10+
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api) (`~/.kaggle/kaggle.json`)
- [DeepSeek API key](https://platform.deepseek.com/api_keys)
- `pip install -e ".[ml]"` (includes LightGBM, XGBoost, CatBoost)

## Roadmap / 路线图

- [x] Phase 0: Project skeleton + DeepSeek validation / 项目骨架 + DeepSeek 验证
- [x] Phase 1: Research Agent (data profile + notebook research + SPEC.md) / 调研 Agent
- [x] Phase 2: Baseline Agent (script generation + execution + experiment DB) / Baseline Agent
- [x] Phase 3: Evaluate + Suggest (strategy advisor with conditional edges) / 评估+建议
- [x] Phase 4: Kaggle Kernel Agent (pull/push/monitor with metadata validation) / Kernel Agent
- [x] Phase 5: Semi-Auto Submit (validator + Human Gate) / 半自动提交
- [x] Phase 6: Advanced (Optuna tuning, ensemble blending) / 高级功能

## License / 许可证

MIT
