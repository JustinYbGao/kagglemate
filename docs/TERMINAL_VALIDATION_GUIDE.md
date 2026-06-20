# KaggleMate Terminal Validation Guide

This guide shows how to validate KaggleMate from a fresh terminal using the
`km` command.  The default offline path does **not** require Kaggle API
credentials, LLM API keys, LangChain, or OpenAI.

---

## Requirements

* Python 3.10+
* A terminal with `bash` / `zsh`
* (Optional) `git` if cloning from GitHub

---

## Installation

### 1. Offline evaluation + tutoring

```bash
cd kagglemate
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -e ".[dev]"
```

What you get:

* `km` CLI entry point
* Unit / integration tests
* Offline `km tutor` / `km ask`
* `km benchmark --synthetic --dry-run`
* No LangChain / OpenAI / LLM API key required

### 2. Full tabular baseline (including training)

Add the ML models so generated training scripts can execute:

```bash
python -m pip install -e ".[ml]"
```

### 3. Conversational LLM agent (optional)

```bash
python -m pip install -e ".[llm]"
# Then configure LLM_API_KEY in .env
cp .env.example .env
```

---

## Quick km sanity checks

```bash
km --help
km tutor --help
km benchmark --help
```

Expected: help text appears, including `tutor`, `ask`, `benchmark`, and
`chat` commands.

### Default `km` without subcommands

```bash
km
```

* With `[llm]` installed: starts the conversational agent.
* Without `[llm]` installed: prints a clear message pointing to `.[llm]` and
  suggests offline commands.

---

## Offline tutoring examples

### Concept tutor

```bash
km tutor "Why does target encoding risk leakage?" \
  --mode concept_tutor \
  --offline
```

Expected output:

* Answer grounded in `docs/ml_concepts/target_encoding.md`
* Mentions out-of-fold (OOF) target encoding
* Includes interpretation, uncertainty, and next verifiable experiment
* No network call

### CV strategy explanation

```bash
km tutor "Why are we using StratifiedKFold for Titanic?" \
  --mode grounded_explanation \
  --competition titanic \
  --offline
```

Expected output:

* References existing `cv_config.json` / `CV_PLAN.md` artifacts
* Explains binary classification / class balance rationale
* Marks uncertainty if detailed rationale is missing

### Experiment diagnosis

```bash
km tutor "Which experiment has the best CV score, and what can we conclude?" \
  --mode experiment_diagnosis \
  --competition titanic \
  --offline
```

Expected output:

* Lists experiments from `experiments.db` / `benchmark_result.json`
* Identifies best CV score without inventing LB scores
* If only one experiment exists, states pairwise comparison is impossible
* Suggests next verifiable experiment

### Show sources

```bash
km tutor "target encoding leakage" \
  --mode concept_tutor \
  --offline \
  --show-sources
```

Expected output:

* Answer panel plus a `Sources:` list with artifact file paths

### `ask` alias

```bash
km ask "Why does target encoding risk leakage?" --mode concept_tutor --offline
```

Same as `km tutor`.

---

## Synthetic benchmark examples

### Dry-run (no ML models required)

```bash
km benchmark --competition titanic --synthetic --dry-run
km benchmark --all --synthetic --dry-run
```

Expected output:

```text
✅ titanic: workflow_completed=True, valid_submission=True
```

Artifacts generated:

* `benchmarks/results/<slug>/<timestamp>/data_profile.json`
* `CV_PLAN.md`, `cv_config.json`
* `experiment_config.json`
* `train_baseline_*.py`
* `strategy_validation_report.json`
* `benchmark_result.json`

### Full offline benchmark (requires `[ml]`)

```bash
km benchmark --competition titanic --synthetic
km benchmark --all --synthetic
```

Expected output:

```text
[run] CV Score: 0.68000 ± 0.04000 (ACCURACY)
✅ titanic: workflow_completed=True, valid_submission=True
```

No Kaggle API is called; the benchmark uses synthetic fixtures under
`benchmarks/fixtures/`.

### Update reports

```bash
python benchmarks/update_reports.py
```

Regenerates:

* `reports/benchmark_summary.md`
* `reports/failure_cases.md`

---

## Full offline verification command set

```bash
# 1. Install
python -m pip install -e ".[dev,ml]"

# 2. km CLI works
km --help

# 3. Offline tutoring works
km tutor "Why does target encoding risk leakage?" --mode concept_tutor --offline

# 4. Experiment diagnosis works
km tutor "Which experiment has the best CV score?" --mode experiment_diagnosis --competition titanic --offline

# 5. Synthetic benchmark works
km benchmark --competition titanic --synthetic

# 6. Reports update
python benchmarks/update_reports.py

# 7. Tests pass
pytest tests/unit -q
pytest tests/integration -q
```

---

## Expected outputs summary

| Command | Expected result | Requires |
|---|---|---|
| `km --help` | Help text with `tutor`/`ask`/`benchmark` | `.[dev]` |
| `km tutor ... --offline` | Grounded answer, no LLM call | `.[dev]` |
| `km benchmark --all --synthetic --dry-run` | All competitions ✅ | `.[dev]` |
| `km benchmark --competition titanic --synthetic` | Trains + validates | `.[dev,ml]` |
| `pytest tests/unit -q` | All unit tests pass | `.[dev]` |
| `pytest tests/integration -q` | Integration tests pass | `.[dev]` |

---

## Common errors and solutions

### `ModuleNotFoundError: No module named 'kagglemate.cli'`

**Cause:** `pyproject.toml` console script points to a module that is missing
or the package was not reinstalled after `pyproject.toml` changes.

**Fix:**

```bash
python -m pip install -e ".[dev]" --force-reinstall --no-deps
```

### `ModuleNotFoundError: No module named 'openai'` when running `km`

**Cause:** You are running the default conversational agent without the `[llm]`
extra.

**Fix:** Use an offline subcommand, or install LLM dependencies:

```bash
km tutor "your question" --offline
# or
python -m pip install -e ".[llm]"
```

### `ModuleNotFoundError: No module named 'lightgbm'` during benchmark

**Cause:** Non-dry-run benchmark executes a generated training script that
imports LightGBM.

**Fix:**

```bash
python -m pip install -e ".[ml]"
```

### `km check` shows red Kaggle credentials

**Cause:** `km check` verifies Kaggle CLI / credentials for real-data workflows.

**Fix for offline use:** No action needed for `--synthetic` benchmarks or
`km tutor`.  To enable real Kaggle downloads, place `kaggle.json` in
`~/.kaggle/`.

### `km tutor` answer is generic / missing sources

**Cause:** No relevant artifacts exist in the project root yet.

**Fix:** Run a benchmark first to generate reports, or add concept docs under
`docs/ml_concepts/`.

```bash
km benchmark --competition titanic --synthetic --dry-run
km tutor "Why are we using StratifiedKFold for Titanic?" --competition titanic --offline
```

### LLM mode error: `Install with: pip install -e '.[llm]'`

**Cause:** You used `--online` or the conversational agent without installing
LangChain / OpenAI.

**Fix:**

```bash
python -m pip install -e ".[llm]"
```

---

## Notes

* The `km` command is an alias for `kagglemate`, defined in `pyproject.toml`.
* The conversational agent (`km` with no args) is the only command that
  requires `[llm]`.
* All `km tutor` / `km ask` / `km benchmark --synthetic` paths are designed to
  work without Kaggle credentials or LLM API keys.
