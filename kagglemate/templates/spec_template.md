# SPEC.md — {{ competition_name }}
>
> **Competition**: {{ competition_slug }}
> **Generated**: {{ generated_at }}
> **Task Type**: {{ competition_type }}
> **Evaluation Metric**: {{ evaluation_metric }}

---

## 1. Competition Overview

| Field | Value |
|-------|-------|
| Name | {{ competition_name }} |
| Slug | {{ competition_slug }} |
| Type | {{ competition_type }} |
| Metric | {{ evaluation_metric }} |
| Train Rows | {{ train_rows }} |
| Test Rows | {{ test_rows }} |
| Features | {{ feature_count }} |

---

## 2. Task Definition

{% if competition_type == "tabular_classification" %}
This is a **binary/multiclass classification** task.
The goal is to predict `{{ target_col }}` given the feature columns.
{% elif competition_type == "tabular_regression" %}
This is a **regression** task.
The goal is to predict a continuous `{{ target_col }}` value.
{% else %}
Task type: {{ competition_type }}. See data profile for details.
{% endif %}

### Input

- Training data: `train.csv` — {{ train_rows }} rows, {{ feature_count }} features
- Test data: `test.csv` — {{ test_rows }} rows
- Sample submission: `sample_submission.csv`

### Output

- Submission format: CSV with columns `{{ submission_cols }}`
- Required row count: {{ submission_rows }}

---

## 3. Data Summary

### Column Types

| Category | Count | Columns |
|----------|-------|---------|
| Numerical | {{ numerical_count }} | {{ numerical_cols }} |
| Categorical | {{ categorical_count }} | {{ categorical_cols }} |

### Target

{% if target_distribution %}
- `{{ target_col }}` distribution: {{ target_distribution }}
{% else %}
- `{{ target_col }}` — see EDA for distribution.
{% endif %}

### Missing Values

{% if missing_values %}
| Column | Missing % |
|--------|-----------|
{% for col, pct in missing_values.items() %}| `{{ col }}` | {{ pct }}% |
{% endfor %}
{% else %}
No missing values detected. ✓
{% endif %}

### Risks

- [ ] Check train/test distribution shift (use adversarial validation)
- [ ] Check for data leakage (ID-based splits, time-based leaks)
- [ ] Verify submission format matches sample exactly

---

## 4. Evaluation Metric

**{{ evaluation_metric }}**

{% if "auc" in evaluation_metric.lower() or "roc" in evaluation_metric.lower() %}
- AUC-ROC measures ranking quality (0.5 = random, 1.0 = perfect)
- Use **StratifiedKFold** for CV (preserves class balance)
- Threshold doesn't matter for AUC — only ranking order
{% elif "accuracy" in evaluation_metric.lower() %}
- Accuracy = correct / total predictions
- Can be misleading for imbalanced classes — check class distribution
{% elif "rmse" in evaluation_metric.lower() or "mse" in evaluation_metric.lower() %}
- (R)MSE penalizes large errors more than small ones
- Consider log-transform on target for skewed distributions
{% endif %}

---

## 5. Submission Format

```
Columns: {{ submission_cols }}
Total rows required: {{ submission_rows }}
```

### Validation Checklist

- [ ] Row count matches test set ({{ submission_rows }})
- [ ] Column names match sample_submission.csv exactly
- [ ] No NaN or infinite values in prediction column
- [ ] File is a valid CSV (not zipped, unless required by competition)

---

## 6. Public Notebook Findings

{{ notebook_findings }}

### Common Patterns

{{ common_patterns }}

### Recommended Baseline Approach

{{ baseline_recommendation }}

---

## 7. Baseline Plan

### Phase 1: Quick Baseline (Day 0–1)

- **Model**: {{ baseline_model }}
- **Features**: All numerical + label-encoded categoricals
- **CV**: {{ cv_strategy }}
- **Target**: CV score > {{ baseline_target }}

### Phase 2: High-ROI Improvements (Day 2–4)

{% for improvement in high_roi_improvements %}
{{ loop.index }}. {{ improvement }}
{% endfor %}

### Phase 3: Fine-tuning (Day 5+)

- Hyperparameter tuning
- Feature selection / dimensionality reduction
- Ensemble exploration

---

## 8. Technical Constraints

- **Environment**: Local Python 3.12 + Kaggle kernel (if submitting via notebook)
- **GPU**: Not required for tabular baseline
- **Inference time**: Should be < 1 hour on Kaggle kernel
- **Internet**: {{ internet_note }}
- **External data**: {{ external_data_note }}

---

## 9. Experiment Plan

| # | Experiment | Model | Expected CV | Status |
|---|-----------|-------|-------------|--------|
| 1 | Baseline | {{ baseline_model }} | — | ⬜ Pending |
| 2 | Feature engineering | {{ baseline_model }} | — | ⬜ Pending |
| 3 | Target encoding | {{ baseline_model }} | — | ⬜ Pending |
| 4 | Hyperparameter tuning | {{ baseline_model }} | — | ⬜ Pending |
| 5 | Ensemble | Blend | — | ⬜ Pending |

---

## 10. Risks and Guardrails

1. **Public LB overfitting** — Don't trust early LB scores. Wait 4+ hours for stabilization.
2. **CV/LB gap** — If CV >> LB, check for overfitting or data leakage.
3. **Submission limits** — Kaggle has daily submission caps.
4. **Code competition rules** — Verify whether this is a Code Competition (stricter rules).
5. **Test set shift** — Run adversarial validation to detect distribution differences.

---

## 11. Next Actions

1. [ ] Run baseline: `python main.py baseline --competition {{ competition_slug }}`
2. [ ] Review data_profile.md for feature ideas
3. [ ] Read top 3 notebooks in detail
4. [ ] Complete rules checklist
5. [ ] Submit baseline and record LB score
