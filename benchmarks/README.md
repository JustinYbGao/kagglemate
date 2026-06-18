# KaggleMate Benchmark Suite

This directory contains a reproducible benchmark harness for evaluating the
KaggleMate tabular workflow across multiple Kaggle competitions.

## Usage

Run a single competition:

```bash
python benchmarks/run_benchmark.py --competition titanic
```

Run all configured competitions:

```bash
python benchmarks/run_benchmark.py --all
```

Dry-run (generate scripts, do not train):

```bash
python benchmarks/run_benchmark.py --all --dry-run
```

Use LLM for feature strategy (default is deterministic heuristic):

```bash
python benchmarks/run_benchmark.py --competition titanic --use-llm
```

## Configuration

Edit `benchmarks/competitions.yaml` to add or remove competitions. Each entry
requires:

```yaml
- slug: titanic
  name: Titanic - Machine Learning from Disaster
  task_type: binary_classification
  metric: accuracy
  target_column: Survived
  id_column: PassengerId
  expected_submission_columns:
    - PassengerId
    - Survived
```

Supported `task_type` values:
- `binary_classification`
- `multiclass_classification`
- `regression`
- `time_series`

## Data

The runner expects data at `competitions/<slug>/data/raw/`:

```
competitions/titanic/data/raw/
  train.csv
  test.csv
  sample_submission.csv
```

If data is missing, the runner records the failure reason in
`benchmark_result.json` and skips the competition rather than crashing.

## Outputs

Each run creates a timestamped directory:

```
benchmarks/results/<slug>/<YYYYMMDD_HHMMSS>/
  data_profile.json
  cv_plan.md
  cv_config.json
  experiment_config.json
  fold_scores.json
  oof_pred.csv
  submission.csv
  run_log.txt
  strategy_validation_report.json
  submission_validation_report.json
  benchmark_result.json
```

Summary reports are updated in:

```
reports/benchmark_summary.md
reports/failure_cases.md
```

## Evaluation Criteria

A benchmark run is considered successful when:

1. The full workflow completes without errors.
2. A valid submission file is generated and passes schema/value checks.
3. CV score and runtime are recorded.
4. All artifacts are persisted for reproducibility.

Failures are recorded in `benchmark_result.json` and aggregated in the summary
report.
