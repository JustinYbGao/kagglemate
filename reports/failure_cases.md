# Failure Case Evaluation

This document records the robustness guardrails built into KaggleMate. Each case
describes a common Kaggle failure mode, the expected behavior of the validation
layer, and the current status.

## Summary

| Case | Component | Expected Behavior | Status |
|---|---|---|---|
| Invalid Submission Columns | submission_validator.py | Block submission, report column mismatch. | ✅ Implemented |
| Target Leakage | strategy_validator.py | Remove target column from features. | ✅ Implemented |
| Wrong ID Order | submission_validator.py | Block submission, report order mismatch. | ✅ Implemented |
| LLM Generated Nonexistent Feature | strategy_validator.py | Remove non-existent feature or fallback to heuristic. | ✅ Implemented |
| Probability Out of Range | submission_validator.py | Block submission for probability metrics. | ✅ Implemented |
| Duplicate IDs | submission_validator.py | Block submission with duplicate ID values. | ✅ Implemented |
| Multiclass Probability Sum | submission_validator.py | Block submission when probabilities do not sum to 1. | ✅ Implemented |
| Regression Extreme Outliers | submission_validator.py | Warn when predictions far exceed training target range. | ✅ Implemented |
| Constant / All-Null Features | strategy_validator.py | Remove uninformative columns. | ✅ Implemented |
| High-Cardinality Categorical | strategy_validator.py | Warn against direct one-hot encoding. | ✅ Implemented |

---

## Invalid Submission Columns

**Trigger:** Submission CSV columns differ from `sample_submission.csv`.

**Expected behavior:**
- `submission_validator.validate()` returns `is_valid=False`.
- `columns_match_sample` check fails with details on missing/extra columns.

**Verification:**
```bash
pytest tests/unit/test_submission_validator.py::test_wrong_columns -q
```

---

## Target Leakage

**Trigger:** The LLM includes the target column (e.g. `Survived`, `SalePrice`) in `src_feature_cols`.

**Expected behavior:**
- `strategy_validator.validate_and_fix()` removes the target from all feature lists.
- An issue of type `target_leakage` is recorded in `strategy_validation_report.json`.

**Verification:**
```bash
pytest tests/unit/test_strategy_validator.py::test_target_column_removed_from_features -q
```

---

## Wrong ID Order

**Trigger:** Submission rows are sorted differently from `sample_submission.csv`.

**Expected behavior:**
- `submission_validator.validate()` returns `is_valid=False`.
- `row_order_matches` check fails.

**Verification:**
```bash
pytest tests/unit/test_submission_validator.py::test_wrong_id_order -q
```

---

## LLM Generated Nonexistent Feature

**Trigger:** The LLM references a column that does not exist in `train.csv`.

**Expected behavior:**
- `strategy_validator.validate_and_fix()` removes the non-existent column.
- If no valid features remain, the strategy falls back to the heuristic baseline.

**Verification:**
```bash
pytest tests/unit/test_strategy_validator.py::test_nonexistent_feature_column_removed -q
pytest tests/unit/test_strategy_validator.py::test_fallback_to_heuristic_when_no_features -q
```

---

## Probability Out of Range

**Trigger:** For `auc` / `logloss` metrics, predictions are outside `[0, 1]`.

**Expected behavior:**
- `submission_validator.validate()` returns `is_valid=False`.
- `probability_range` check reports the offending range.

**Verification:**
```bash
pytest tests/unit/test_submission_validator.py::test_probability_out_of_range -q
```

---

## Duplicate IDs

**Trigger:** The submission contains repeated ID values.

**Expected behavior:**
- `submission_validator.validate()` returns `is_valid=False`.
- `no_duplicate_ids` check reports the count.

**Verification:**
```bash
pytest tests/unit/test_submission_validator.py::test_duplicate_ids -q
```

---

## Multiclass Probability Sum

**Trigger:** A multiclass submission has probability columns that do not sum to ~1 per row.

**Expected behavior:**
- `submission_validator.validate()` returns `is_valid=False`.
- `probability_row_sum` check reports the actual row-sum range.

**Verification:**
```bash
pytest tests/unit/test_submission_validator.py::test_multiclass_probability_sum -q
```

---

## Regression Extreme Outliers

**Trigger:** Regression predictions exceed the training target range by a large margin.

**Expected behavior:**
- `submission_validator.validate()` returns `is_valid=True` but adds a warning.
- `regression_outliers` check reports the count of out-of-range values.

**Verification:**
```bash
pytest tests/unit/test_submission_validator.py::test_regression_outlier_warning -q
```

---

## Constant / All-Null Features

**Trigger:** A feature has only one unique value or is nearly all missing.

**Expected behavior:**
- `strategy_validator.validate_and_fix()` removes the column.
- Issues of type `constant_column` or `all_null_column` are recorded.

**Verification:**
Unit test coverage added in `tests/unit/test_strategy_validator.py`.

---

## High-Cardinality Categorical

**Trigger:** A categorical column has more than 50 unique values and is likely to be one-hot encoded.

**Expected behavior:**
- `strategy_validator.validate_and_fix()` emits a warning.
- Issue of type `high_cardinality` is recorded with action `warn_only`.

**Verification:**
```bash
pytest tests/unit/test_strategy_validator.py::test_high_cardinality_categorical_warning -q
```
