# SPEC.md / 竞赛策略文档 — {{ competition_name }}
>
> **Competition / 比赛**: {{ competition_slug }}
> **Generated / 生成时间**: {{ generated_at }}
> **Task Type / 任务类型**: {{ competition_type }}
> **Evaluation Metric / 评价指标**: {{ evaluation_metric }}

---

## 1. Competition Overview / 比赛概览

| Field / 字段 | Value / 值 |
|-------|-------|
| Name / 名称 | {{ competition_name }} |
| Slug / 标识 | {{ competition_slug }} |
| Type / 类型 | {{ competition_type }} |
| Metric / 指标 | {{ evaluation_metric }} |
| Train Rows / 训练集行数 | {{ train_rows }} |
| Test Rows / 测试集行数 | {{ test_rows }} |
| Features / 特征数 | {{ feature_count }} |

---

## 2. Task Definition / 任务定义

{% if competition_type == "tabular_classification" %}
This is a **binary/multiclass classification** task. / 这是一个**二分类/多分类**任务。
The goal is to predict `{{ target_col }}` given the feature columns. / 目标是根据特征列预测 `{{ target_col }}`。
{% elif competition_type == "tabular_regression" %}
This is a **regression** task. / 这是一个**回归**任务。
The goal is to predict a continuous `{{ target_col }}` value. / 目标是预测连续的 `{{ target_col }}` 值。
{% else %}
Task type / 任务类型: {{ competition_type }}. See data profile for details. / 详见数据报告。
{% endif %}

### Input / 输入

- Training data / 训练数据: `train.csv` — {{ train_rows }} rows / 行, {{ feature_count }} features / 特征
- Test data / 测试数据: `test.csv` — {{ test_rows }} rows / 行
- Sample submission / 示例提交: `sample_submission.csv`

### Output / 输出

- Submission format / 提交格式: CSV with columns / 列: `{{ submission_cols }}`
- Required row count / 所需行数: {{ submission_rows }}

---

## 3. Data Summary / 数据摘要

### Column Types / 列类型

| Category / 类别 | Count / 数量 | Columns / 列名 |
|----------|-------|---------|
| Numerical / 数值型 | {{ numerical_count }} | {{ numerical_cols }} |
| Categorical / 类别型 | {{ categorical_count }} | {{ categorical_cols }} |

### Target / 目标变量

{% if target_distribution %}
- `{{ target_col }}` distribution / 分布: {{ target_distribution }}
{% else %}
- `{{ target_col }}` — see EDA for distribution. / 分布详见 EDA。
{% endif %}

### Missing Values / 缺失值

{% if missing_values %}
| Column / 列名 | Missing % / 缺失比例 |
|--------|-----------|
{% for col, pct in missing_values.items() %}| `{{ col }}` | {{ pct }}% |
{% endfor %}
{% else %}
No missing values detected. ✓ / 未检测到缺失值。✓
{% endif %}

### Risks / 风险

- [ ] Check train/test distribution shift (use adversarial validation) / 检查训练/测试集分布偏移（建议对抗验证）
- [ ] Check for data leakage (ID-based splits, time-based leaks) / 检查数据泄露（ID拆分、时间泄露等）
- [ ] Verify submission format matches sample exactly / 确认提交格式与示例完全一致

---

## 4. Evaluation Metric / 评价指标

**{{ evaluation_metric }}**

{% if "auc" in evaluation_metric.lower() or "roc" in evaluation_metric.lower() %}
- AUC-ROC measures ranking quality (0.5 = random, 1.0 = perfect) / AUC-ROC 衡量排序质量（0.5=随机，1.0=完美）
- Use **StratifiedKFold** for CV (preserves class balance) / 使用 StratifiedKFold 验证（保持类别平衡）
- Threshold doesn't matter for AUC — only ranking order / AUC 不依赖阈值，只关注排序
{% elif "accuracy" in evaluation_metric.lower() %}
- Accuracy = correct / total predictions / 准确率 = 正确预测数 / 总数
- Can be misleading for imbalanced classes — check class distribution / 类别不均衡时可能误导——请检查类别分布
{% elif "rmse" in evaluation_metric.lower() or "mse" in evaluation_metric.lower() %}
- (R)MSE penalizes large errors more than small ones / RMSE 对大误差的惩罚远大于小误差
- Consider log-transform on target for skewed distributions / 对于偏态分布，建议对目标值做对数变换
{% endif %}

---

## 5. Submission Format / 提交格式

```
Columns / 列: {{ submission_cols }}
Total rows required / 所需行数: {{ submission_rows }}
```

### Validation Checklist / 验证清单

- [ ] Row count matches test set ({{ submission_rows }}) / 行数与测试集匹配
- [ ] Column names match sample_submission.csv exactly / 列名与示例提交完全一致
- [ ] No NaN or infinite values in prediction column / 预测列无 NaN 或无穷值
- [ ] File is a valid CSV (not zipped, unless required by competition) / 文件为有效 CSV（除非比赛要求压缩）

---

## 6. Public Notebook Findings / 公开 Notebook 调研结果

{{ notebook_findings }}

### Common Patterns / 共性模式

{{ common_patterns }}

### Recommended Baseline Approach / 推荐 Baseline 方案

{{ baseline_recommendation }}

---

## 7. Baseline Plan / Baseline 计划

### Phase 1: Quick Baseline / 快速基线 (Day / 第 0–1 天)

- **Model / 模型**: {{ baseline_model }}
- **Features / 特征**: All numerical + label-encoded categoricals / 全部数值型 + 标签编码的类别型
- **CV / 验证**: {{ cv_strategy }}
- **Target / 目标**: CV score > {{ baseline_target }}

### Phase 2: High-ROI Improvements / 高投入产出比改进 (Day / 第 2–4 天)

{% for improvement in high_roi_improvements %}
{{ loop.index }}. {{ improvement }}
{% endfor %}

### Phase 3: Fine-tuning / 精细调优 (Day / 第 5 天起)

- Hyperparameter tuning / 超参数调优
- Feature selection / dimensionality reduction / 特征选择/降维
- Ensemble exploration / 模型集成探索

---

## 8. Technical Constraints / 技术约束

- **Environment / 环境**: Local Python 3.12 + Kaggle kernel (if submitting via notebook) / 本地 Python 3.12 + Kaggle kernel（如通过 notebook 提交）
- **GPU**: Not required for tabular baseline / Tabular baseline 不需要 GPU
- **Inference time / 推理时间**: Should be < 1 hour on Kaggle kernel / 应在 Kaggle kernel 上 1 小时内完成
- **Internet / 联网**: {{ internet_note }}
- **External data / 外部数据**: {{ external_data_note }}

---

## 9. Experiment Plan / 实验计划

| # | Experiment / 实验 | Model / 模型 | Expected CV / 预期 CV | Status / 状态 |
|---|-----------|-------|-------------|--------|
| 1 | Baseline / 基线 | {{ baseline_model }} | — | ⬜ Pending / 待做 |
| 2 | Feature engineering / 特征工程 | {{ baseline_model }} | — | ⬜ Pending / 待做 |
| 3 | Target encoding / 目标编码 | {{ baseline_model }} | — | ⬜ Pending / 待做 |
| 4 | Hyperparameter tuning / 超参数调优 | {{ baseline_model }} | — | ⬜ Pending / 待做 |
| 5 | Ensemble / 模型集成 | Blend | — | ⬜ Pending / 待做 |

---

## 10. Risks and Guardrails / 风险与防护

1. **Public LB overfitting / 公开榜过拟合** — Don't trust early LB scores. Wait 4+ hours for stabilization. / 不要轻信早期 LB 分数，等待 4 小时以上稳定后再判断。
2. **CV/LB gap / CV/LB 差距** — If CV >> LB, check for overfitting or data leakage. / 如果 CV 大幅优于 LB，检查是否过拟合或数据泄露。
3. **Submission limits / 提交次数限制** — Kaggle has daily submission caps. / Kaggle 每日提交次数有限。
4. **Code competition rules / 代码竞赛规则** — Verify whether this is a Code Competition (stricter rules). / 确认是否为代码竞赛（规则更严格）。
5. **Test set shift / 测试集偏移** — Run adversarial validation to detect distribution differences. / 通过对抗验证检测分布差异。

---

## 11. Next Actions / 下一步行动

1. [ ] Run baseline / 运行基线: `python main.py baseline --competition {{ competition_slug }}`
2. [ ] Review data_profile.md for feature ideas / 查看 data_profile.md 寻找特征灵感
3. [ ] Read top 3 notebooks in detail / 详细阅读 Top 3 notebook
4. [ ] Complete rules checklist / 完成规则检查清单
5. [ ] Submit baseline and record LB score / 提交基线并记录 LB 分数
