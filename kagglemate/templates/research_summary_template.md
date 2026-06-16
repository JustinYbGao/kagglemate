# Research Summary / 调研报告 — {{ competition_name }}
>
> **Competition / 比赛**: {{ competition_slug }}
> **Generated / 生成时间**: {{ generated_at }}
> **Notebooks analyzed / 分析 Notebook 数**: {{ notebook_count }}

---

## Top Public Approaches / 公开高分方案

{% for nb in notebooks %}
### {{ loop.index }}. {{ nb.title }}
- **Author / 作者**: {{ nb.author }}
- **Votes / 票数**: {{ nb.votes }}
- **Model / 模型**: {{ nb.model }}
{% if nb.lb_score %}- **LB Score / 榜单分数**: {{ nb.lb_score }}{% endif %}
- **CV Method / 验证方法**: {{ nb.cv_method }}
- **Key Techniques / 关键技术**: {{ nb.key_techniques | join(", ") }}
- **Worth Reproducing / 值得复现**: {{ "是 Yes" if nb.worth_reproducing else "否 No" }}
- **Notes / 备注**: {{ nb.notes }}

{% endfor %}

---

## Common Patterns Across Top Notebooks / 高分方案共性

{{ common_patterns }}

---

## Potential Baseline / 推荐 Baseline 方案

Based on the most common successful approach / 基于最常见的高分方案：

- **Model / 模型**: {{ recommended_model }}
- **CV / 验证策略**: {{ recommended_cv }}
- **Feature engineering / 特征工程**: {{ recommended_fe }}

---

## High-ROI Improvement Ideas / 高投入产出比改进方向

{% for idea in improvement_ideas %}
{{ loop.index }}. {{ idea }}
{% endfor %}

---

## Notebooks to Study in Detail / 建议深入研究的 Notebook

{% for nb in notebooks_to_study %}- [**{{ nb.ref }}**](https://www.kaggle.com/code/{{ nb.ref }}) — {{ nb.reason }}
{% endfor %}
