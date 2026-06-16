# KaggleMate

> **LangGraph-based Kaggle competition assistant agent powered by DeepSeek V4**

Takes a Kaggle competition slug and autonomously researches public notebooks, profiles data, generates strategy documents, writes and runs baseline ML models, tracks experiments, and recommends next steps — all with a Human Gate for submissions.

[![Python](https://img.shields.io/badge/Python-3.10%2B-blue)](https://python.org)
[![LangGraph](https://img.shields.io/badge/LangGraph-1.2-green)](https://langchain.com/langgraph)
[![DeepSeek](https://img.shields.io/badge/DeepSeek-V4%20Pro-purple)](https://platform.deepseek.com)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

---

## Quick Start

```bash
# 1. Clone & install
git clone https://github.com/JustinYbGao/kagglemate.git
cd kagglemate
pip install -e ".[ml]"

# 2. Configure
cp .env.example .env
# Edit .env with your DEEPSEEK_API_KEY

# 3. Verify
python main.py check

# 4. Run research + baseline on any competition
python main.py research titanic
python main.py baseline titanic
python main.py run titanic
```

## Architecture

```
LangGraph StateGraph (11 nodes):
  
  Init ──▶ Analyze ──▶ Research ──▶ Plan ──▶ Baseline ──▶ Run ──▶ Evaluate
                                                                      │
                                                              ┌───────┴───────┐
                                                              ▼               ▼
                                                           Suggest ──▶      END
                                                              │         (stop)
                                                              ▼
                                                            END
```

### Nodes

| Node | Phase | What it does |
|------|-------|-------------|
| `init` | 1 | Downloads data, creates workspace |
| `analyze` | 1 | Profiles CSV data, identifies task type & metric |
| `research` | 1 | Fetches top public notebooks, LLM summarizes techniques |
| `plan` | 1 | Generates SPEC.md + research_summary.md + rules_checklist.md |
| `baseline` | 2 | LLM designs features + Jinja2 renders training script |
| `run` | 2 | Executes script, parses CV score, saves to experiments.db |
| `evaluate` | 3 | CV/LB gap analysis, overfitting detection, plateau checks |
| `suggest` | 3 | LLM recommends next 3-5 experiments with expected improvement |
| `kernel` | 4 | Pull/push/monitor Kaggle kernels with metadata validation |
| `submit` | 5 | Submission validation + Human Gate confirmation |

### Conditional Edge

```
Evaluate ──▶ _should_continue=True ──▶ Suggest ──▶ END
          ──▶ _should_continue=False ──▶ END (≥3 consecutive failures)
```

## CLI Commands

```bash
# ── Research ──
python main.py research <slug>        # Full pipeline: data + notebooks + SPEC
python main.py profile <slug>         # Data profiling only
python main.py spec <slug>            # Generate SPEC.md from existing data

# ── Modeling ──
python main.py baseline <slug>        # Generate training script
python main.py run <slug>             # Execute + record experiment
python main.py suggest <slug>         # Get next-step recommendations

# ── Experiments ──
python main.py experiments <slug> --action list
python main.py experiments <slug> --action show --id <id>
python main.py experiments <slug> --action compare --ids 1,2,3
python main.py experiments <slug> --action log-lb --id <id> --lb <score>

# ── Kaggle Kernels ──
python main.py notebook pull <ref> -c <slug>
python main.py kernel push <dir> -c <slug>
python main.py kernel monitor <ref>

# ── Submissions (always requires human "YES") ──
python main.py submission validate -c <slug> -f <file>.csv
python main.py submission submit -c <slug> -f <file>.csv -m "message"
python main.py submission status -c <slug>
```

## Example: Titanic

```bash
$ python main.py research titanic
  [init] → init
  [analyze] → analyze
  [research] → research  
  [plan] → plan
✅ Research complete!
  📄 SPEC.md (8.0 KB)
  📄 data_profile.md (1.2 KB)
  📄 rules_checklist.md (2.2 KB)
  Task type: tabular_classification
  Metric: accuracy

$ python main.py baseline titanic
  Model: LightGBM
  Features: 11 selected (LLM-chosen feature engineering)

$ python main.py run titanic
  CV: 0.8473 ± 0.0149 (ACCURACY)
  Top features: Fare, Age, Pclass, FamilySize, Title...
  ✅ Experiment #1 saved

$ python main.py suggest titanic
  💡 5 recommendations generated:
  1. LightGBM regularization (+0.010)
  2. Feature: Title + FamilySize (+0.015)
  3. StratifiedKFold fix (+0.000)
  4. Ensemble LightGBM + LogisticRegression (+0.005)
  5. Fare/Age binning (+0.005)
```

## Key Design Decisions

| Decision | Choice | Why |
|----------|--------|-----|
| Agent framework | LangGraph Graph API | Explicit state machine, conditional edges, visualization |
| LLM provider | DeepSeek V4 Pro | 1M context, strong tool calling, affordable |
| Code generation | Template + LLM fill | Training scripts are Jinja2 templates; LLM only writes feature engineering |
| Submission | CLI Human Gate | Never auto-submit. User must type YES after validation. |
| Experiment tracking | SQLite per competition | Portable, zero-config, easy to backup |
| Data format support | Tabular CSV first | Most reliable format for automation |

## Project Structure

```
kagglemate/
├── graph/           # LangGraph StateGraph nodes & builder
│   ├── builder.py   # Graph assembly + conditional edges
│   ├── state.py     # KaggleAgentState TypedDict
│   └── nodes/       # 10 node implementations
├── tools/           # Domain tools (wrappers around Kaggle CLI, pandas)
├── memory/          # Experiment tracking (SQLite)
├── templates/       # Jinja2 templates (SPEC, baseline scripts, reports)
├── main.py          # Typer CLI entry point
└── tests/           # DeepSeek tool calling verification
```

## Prerequisites

- Python 3.10+
- [Kaggle API credentials](https://github.com/Kaggle/kaggle-api) (`~/.kaggle/kaggle.json`)
- [DeepSeek API key](https://platform.deepseek.com/api_keys)
- `pip install -e ".[ml]"` (includes LightGBM, XGBoost, CatBoost)

## Roadmap

- [x] Phase 0: Project skeleton + DeepSeek validation
- [x] Phase 1: Research Agent (data profile + notebook research + SPEC.md)
- [x] Phase 2: Baseline Agent (script generation + execution + experiment DB)
- [x] Phase 3: Evaluate + Suggest (strategy advisor with conditional edges)
- [x] Phase 4: Kaggle Kernel Agent (pull/push/monitor with metadata validation)
- [x] Phase 5: Semi-Auto Submit (validator + Human Gate)
- [ ] Phase 6: Advanced (Optuna tuning, ensemble, multi-agent, non-tabular data)

## License

MIT
