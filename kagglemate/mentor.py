"""Mentor Layer — 竞赛导师模式 / Competition Mentorship Engine

KaggleMate is not just a tool — it's a teacher. This module provides:
1. explain_notebook: pull a notebook and walk through the code, explaining ML concepts
2. explain_concept: explain any ML concept in the context of the current competition
3. compare_approaches: compare two experiments and explain why scores differ

Design principle: The user should UNDERSTAND, not just execute.
After every automated action, explain WHY it works and WHAT was learned.
"""

from __future__ import annotations

import json
from pathlib import Path

from kagglemate.tools.llm_client import simple_prompt
from kagglemate.config import config


# ═══════════════════════════════════════════════════════════════════════════════
# System Prompt — The Mentor Persona / 导师人设
# ═══════════════════════════════════════════════════════════════════════════════

MENTOR_SYSTEM_PROMPT_TEMPLATE = """You are KaggleMate, a Kaggle competition **mentor**. Your primary goal is not just to execute tasks, but to help the user **learn** and **improve** as a data scientist through each competition.

## Your Identity
- You are a patient, knowledgeable mentor who explains the "why" behind every action
- You teach ML concepts when the opportunity arises
- You walk through code and explain what each part does
- You help the user understand other competitors' thinking by analyzing their notebooks
- You celebrate improvements and help debug failures constructively

## User Info
- Kaggle username: {kaggle_username}
- Kaggle API credentials: configured (at ~/.kaggle/kaggle.json)

## Your Capabilities (Tool List)
You have {tool_count} tools available. Use them to research, build, tune, submit — and explain everything along the way.

## HOW TO BEHAVE / 行为准则

### 1. EXPLAIN BEFORE YOU ACT
Before running research, building a model, or tuning: tell the user WHAT you're about to do and WHY. One sentence is enough.

### 2. TEACH AFTER YOU ACT
After every experiment completes: point out ONE thing the user can learn from the result.
- "CV 提高了 → 说明特征工程方向对了，具体是因为..."
- "CV 下降了 → 可能过拟合了，原因是参数 X 设置太大..."
- "分数没变化 → 说明这个方向已经到天花板了，下次应该尝试..."

### 3. WHEN EXPLAINING NOTEBOOKS
When the user asks to study a notebook:
- Pull it with metadata preserved
- Walk through the code section by section
- For each section: explain what it does, why the author chose this approach, what ML concept it illustrates
- Point out clever tricks and potential pitfalls
- Relate it to the user's current competition

### 4. WHEN EXPLAINING CONCEPTS
When the user asks "what is X?" or "why do people use Y?":
- Explain in plain language first, then with ML terminology
- Give a concrete example from the current competition's data
- Mention when to use it and when NOT to use it
- If relevant, offer to generate a small experiment to demonstrate

### 5. COMPETITION CONTEXT
- Track which competition the user is working on
- Remember what experiments have been tried
- When giving advice, reference SPECIFIC experiments and results, not generalities

### 6. NEVER BLINDLY EXECUTE
- If the user's request doesn't make sense for their competition type, explain why and suggest alternatives
- If a submission might waste a daily slot, warn the user with specific reasons

### 7. USE THE USER'S LANGUAGE
- Respond in Chinese if the user speaks Chinese. 用中文回复。
- Use technical terms in English when they have no good Chinese translation

## The Learning Loop / 学习循环

```
Analyze → Understand → Experiment → Reflect → Improve
   ↑                                          │
   └──────────────────────────────────────────┘
```

At each step, your job is to help the user move through this loop:
1. **Analyze**: Research the competition, study notebooks, understand the data
2. **Understand**: Explain concepts, walk through code, answer "why" questions
3. **Experiment**: Build models, tune parameters, try ideas
4. **Reflect**: After each experiment — what worked? what didn't? what did we learn?
5. **Improve**: Apply the lesson to the next experiment
"""


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 1: explain_notebook / 讲解 Notebook
# ═══════════════════════════════════════════════════════════════════════════════

EXPLAIN_NOTEBOOK_PROMPT = """You are a patient Kaggle mentor. Walk through the following notebook code and explain it section by section.

## Competition Context
- Competition: {competition_name}
- Type: {competition_type}
- Target: {target_col}
- Metric: {evaluation_metric}

## Notebook: {notebook_title} by {notebook_author}
{votes_text}

## Notebook Code
```python
{notebook_code}
```

## Instructions
Walk through this notebook section by section. For each logical section, explain:

1. **What this section does** (一句话概括)
2. **Why the author chose this approach** (作者的思路)
3. **Key ML concept illustrated** (涉及的机器学习概念)
4. **What you can learn from it** (你可以学到什么)
5. **Potential improvements or pitfalls** (改进空间或潜在问题)

Format your response as a structured walkthrough. Use Chinese if the user speaks Chinese.
After the walkthrough, end with a "Key Takeaways / 要点总结" section (3-5 bullets).

Keep each section explanation to 3-5 sentences. Be specific, not generic."""


def explain_notebook(competition_slug: str, kernel_ref: str,
                     competition_name: str = "", competition_type: str = "",
                     target_col: str = "", evaluation_metric: str = "",
                     max_code_lines: int = 500) -> str:
    """Pull a notebook and generate a line-by-line code walkthrough.

    Args:
        competition_slug: e.g. 'titanic'
        kernel_ref: e.g. 'alexisbcook/titanic-tutorial'
        competition_name, competition_type, target_col, evaluation_metric:
            Competition context for the explanation.
        max_code_lines: Max lines of code to send to LLM (truncate if longer).

    Returns:
        Markdown walkthrough of the notebook.
    """
    from kagglemate.tools.kaggle_cli import KaggleCLI

    # ── Step 1: Pull notebook ──
    comp_dir = config.COMPETITIONS_DIR / competition_slug
    nb_dir = comp_dir / "notebooks" / kernel_ref.replace("/", "_")
    nb_dir.mkdir(parents=True, exist_ok=True)

    try:
        KaggleCLI.pull_kernel(kernel_ref, nb_dir)
    except RuntimeError as e:
        return f"无法拉取 Notebook `{kernel_ref}`: {e}"

    # ── Step 2: Read notebook content ──
    nb_files = list(nb_dir.glob("*.ipynb"))
    if not nb_files:
        return f"Notebook 已拉取到 `{nb_dir}`，但未找到 .ipynb 文件。"

    try:
        import nbformat
        nb = nbformat.read(str(nb_files[0]), as_version=4)
    except Exception as e:
        return f"无法解析 Notebook: {e}"

    # ── Step 3: Extract code cells ──
    code_lines = []
    for cell in nb.cells:
        if cell.cell_type == "code" and cell.source.strip():
            code_lines.append(cell.source.strip())

    full_code = "\n\n".join(code_lines)
    total_lines = full_code.count("\n") + 1

    if total_lines > max_code_lines:
        full_code = full_code[:max_code_lines * 80]  # rough char estimate
        full_code += f"\n\n# ... (代码被截断，共 {total_lines} 行，显示了前 {max_code_lines} 行)"

    # ── Step 4: Read metadata ──
    metadata_path = nb_dir / "kernel-metadata.json"
    notebook_title = kernel_ref
    notebook_author = ""
    votes_text = ""
    if metadata_path.exists():
        try:
            meta = json.loads(metadata_path.read_text())
            notebook_title = meta.get("title", kernel_ref)
            notebook_author = meta.get("author", "") or kernel_ref.split("/")[0]
        except Exception:
            pass

    # ── Step 5: Generate explanation ──
    prompt = EXPLAIN_NOTEBOOK_PROMPT.format(
        competition_name=competition_name or competition_slug,
        competition_type=competition_type or "unknown",
        target_col=target_col or "unknown",
        evaluation_metric=evaluation_metric or "unknown",
        notebook_title=notebook_title,
        notebook_author=notebook_author,
        votes_text=votes_text,
        notebook_code=full_code,
    )

    print("  [mentor] 正在分析 Notebook 代码...")
    try:
        explanation = simple_prompt(prompt)
    except Exception:
        explanation = (
            f"## Notebook: {notebook_title}\n\n"
            f"代码已拉取到 `{nb_dir}`。LLM 分析暂时不可用，请手动查看。\n\n"
            f"**文件列表**:\n"
            + "\n".join(f"- {f.name}" for f in sorted(nb_dir.iterdir()))
        )

    # ── Step 6: Return with file path ──
    header = (
        f"## 📖 Notebook 讲解: {notebook_title}\n"
        f"**作者**: {notebook_author} | **文件**: `{nb_dir}`\n"
        f"**代码行数**: {total_lines}\n\n"
        f"---\n\n"
    )
    return header + explanation


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 2: explain_concept / 讲解概念
# ═══════════════════════════════════════════════════════════════════════════════

EXPLAIN_CONCEPT_PROMPT = """You are a patient Kaggle mentor. Explain the following ML concept clearly and thoroughly.

## Competition Context (for examples)
- Competition: {competition_name}
- Type: {competition_type}
- Target: {target_col} (distribution: {target_distribution})
- Metric: {evaluation_metric}
- Train shape: {train_shape}
- Key features: {key_features}

## Concept to Explain
{concept}

## Instructions
1. **What is it? / 是什么？** (1-2 sentences, plain language)
2. **How does it work? / 原理** (2-3 sentences, with ML terminology)
3. **Example from THIS competition / 在这个比赛中的应用** — give a concrete code example using the actual columns from this dataset
4. **When to use it / 什么时候用** (when it works well)
5. **When NOT to use it / 什么时候不要用** (common pitfalls)
6. **Related concepts / 相关概念** (2-3 related techniques the user should know)

Use Chinese if the user speaks Chinese. Be specific, not generic. Use actual column names from the data."""


def explain_concept(concept: str, competition_slug: str,
                    competition_name: str = "", competition_type: str = "",
                    target_col: str = "", evaluation_metric: str = "",
                    target_distribution: str = "", train_shape: str = "",
                    key_features: str = "") -> str:
    """Explain an ML concept with competition-specific examples.

    If competition data is available, the explanation will use actual
    column names, shapes, and context to make it concrete.
    """
    # Try to load competition context
    if not train_shape:
        data_dir = config.COMPETITIONS_DIR / competition_slug / "data" / "raw"
        if data_dir.exists():
            csvs = list(data_dir.glob("train*.csv"))
            if csvs:
                try:
                    import pandas as pd
                    df = pd.read_csv(csvs[0], nrows=5)
                    train_shape = f"{'?'} rows × {len(df.columns)} cols"
                    if not key_features:
                        key_features = ", ".join(df.columns[:8].tolist())
                except Exception:
                    pass

    prompt = EXPLAIN_CONCEPT_PROMPT.format(
        competition_name=competition_name or competition_slug,
        competition_type=competition_type or "unknown",
        target_col=target_col or "unknown",
        target_distribution=target_distribution or "unknown",
        evaluation_metric=evaluation_metric or "unknown",
        train_shape=train_shape or "unknown",
        key_features=key_features or "unknown",
        concept=concept,
    )

    print(f"  [mentor] 正在讲解: {concept}")
    try:
        return simple_prompt(prompt)
    except Exception:
        return f"## {concept}\n\nLLM 暂时不可用。请在 Kaggle 上搜索 '**{concept}**' 了解详情。"


# ═══════════════════════════════════════════════════════════════════════════════
# Tool 3: compare_approaches / 对比方案
# ═══════════════════════════════════════════════════════════════════════════════

COMPARE_APPROACHES_PROMPT = """You are a Kaggle mentor. Compare two experiments and explain why their scores differ.

## Competition Context
- Competition: {competition_name}
- Type: {competition_type}
- Metric: {evaluation_metric}

## Experiment A
- Name: {exp_a_name}
- Model: {exp_a_model}
- CV Score: {exp_a_cv}
- LB Score: {exp_a_lb}
- Features: {exp_a_features}
- Params: {exp_a_params}
- Feature Importance: {exp_a_fi}

## Experiment B
- Name: {exp_b_name}
- Model: {exp_b_model}
- CV Score: {exp_b_cv}
- LB Score: {exp_b_lb}
- Features: {exp_b_features}
- Params: {exp_b_params}
- Feature Importance: {exp_b_fi}

## Instructions
1. **Which is better and by how much? / 哪个更好？差距多大？** (1 sentence)
2. **Why? / 为什么？** — Analyze the differences in features, parameters, and model architecture. What SPECIFIC changes caused the score difference?
3. **What did we learn? / 学到了什么？** — What generalizable lesson can the user take away?
4. **What should we try next? / 下一步？** — Based on this comparison, what's the most promising direction?

Be specific. Reference actual feature names, parameter values, and score differences."""


def compare_approaches(experiment_a: dict, experiment_b: dict,
                       competition_name: str = "", competition_type: str = "",
                       evaluation_metric: str = "") -> str:
    """Compare two experiments and explain the score difference in detail.

    Args:
        experiment_a, experiment_b: Dicts from ExperimentStore.get(id).
    """
    def fmt_exp(exp: dict) -> dict:
        cv = exp.get("cv_score")
        cv_str = f"{cv:.5f}" if cv else "N/A"
        lb = exp.get("lb_score")
        lb_str = f"{lb:.5f}" if lb else "N/A"
        features = exp.get("features") or []
        params = exp.get("params") or {}
        fi = exp.get("feature_importance") or []
        fi_str = ", ".join(f"{n}({v:.2f})" for n, v in fi[:5]) if fi else "N/A"
        return {
            "exp_name": exp.get("experiment_name") or exp.get("name", "?"),
            "exp_model": exp.get("model_name") or exp.get("model", "?"),
            "exp_cv": cv_str,
            "exp_lb": lb_str,
            "exp_features": f"{len(features)} features: {', '.join(features[:10])}",
            "exp_params": json.dumps(params, indent=2) if params else "N/A",
            "exp_fi": fi_str,
        }

    a = fmt_exp(experiment_a)
    b = fmt_exp(experiment_b)

    prompt = COMPARE_APPROACHES_PROMPT.format(
        competition_name=competition_name or "Unknown",
        competition_type=competition_type or "unknown",
        evaluation_metric=evaluation_metric or "unknown",
        exp_a_name=a["exp_name"], exp_a_model=a["exp_model"],
        exp_a_cv=a["exp_cv"], exp_a_lb=a["exp_lb"],
        exp_a_features=a["exp_features"], exp_a_params=a["exp_params"],
        exp_a_fi=a["exp_fi"],
        exp_b_name=b["exp_name"], exp_b_model=b["exp_model"],
        exp_b_cv=b["exp_cv"], exp_b_lb=b["exp_lb"],
        exp_b_features=b["exp_features"], exp_b_params=b["exp_params"],
        exp_b_fi=b["exp_fi"],
    )

    print(f"  [mentor] 对比: {a['exp_name']} vs {b['exp_name']}")
    try:
        return simple_prompt(prompt)
    except Exception:
        return (
            f"## 对比: {a['exp_name']} vs {b['exp_name']}\n\n"
            f"| | {a['exp_name']} | {b['exp_name']} |\n"
            f"|---|---|---|\n"
            f"| CV | {a['exp_cv']} | {b['exp_cv']} |\n"
            f"| LB | {a['exp_lb']} | {b['exp_lb']} |\n"
            f"| Model | {a['exp_model']} | {b['exp_model']} |\n"
            f"| Features | {a['exp_features']} | {b['exp_features']} |\n"
        )
