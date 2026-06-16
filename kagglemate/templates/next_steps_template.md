# Next Steps / 下一步建议 — {{ competition_name }}
>
> **Competition / 比赛**: {{ competition_slug }}
> **Generated / 生成时间**: {{ generated_at }}
> **Total experiments / 总实验数**: {{ total_experiments }}

---

## Current Best / 当前最佳

| Metric / 指标 | Value / 值 |
|--------|-------|
| CV Score / 交叉验证分数 | {{ best_cv }} |
| LB Score / 排行榜分数 | {{ best_lb }} |

---

## Situation Diagnosis / 现状诊断

{{ situation }}

---

## Recommended Next Experiments / 推荐下一步实验

{% for rec in recommendations %}
### {{ loop.index }}. {{ rec.name | replace("_", " ") | title }}
- **Impact / 预期效果**: {{ rec.expected_impact | upper }}
- **Estimated CV improvement / 预估 CV 提升**: {{ rec.cv_improvement_estimate }}
- **What to do / 执行内容**: {{ rec.what_to_do }}
- **Why / 理由**: {{ rec.reason }}
- **Risk / 风险**: {{ rec.risk }}

{% endfor %}

---

## Decision / 决策

Choose ONE experiment from the recommendations above. To implement / 从以上推荐中选择一项实验实施：

```bash
# Edit the baseline script with the chosen technique,
# or run the agent to generate a new experiment:
# 修改 baseline 脚本以应用选定技术，或让 agent 生成新实验：
python main.py baseline --competition {{ competition_slug }}

# Then run it / 然后运行：
python main.py run --competition {{ competition_slug }}

# Then get new suggestions / 然后获取新建议：
python main.py suggest --competition {{ competition_slug }}
```

---

## General Guidance / 通用指导

1. **Don't chase LB directly / 不要直接冲榜单** — optimize CV, not public leaderboard. / 优化 CV，而非公开排行榜。
2. **One change at a time / 一次只改一处** — change one thing per experiment so you know what worked. / 每次实验只改一个变量，才能明确知道什么起了作用。
3. **CV/LB gap is your compass / CV/LB 差距是方向标** — wide gap → reduce complexity. Tight gap → add complexity. / 差距大→降低复杂度；差距小→增加复杂度。
4. **Feature engineering > Hyperparameter tuning / 特征工程优于调参** — new features usually beat better params. / 新特征通常比更优的参数收益更大。
5. **When stuck / 遇到瓶颈时**: go back to notebooks, or try a completely different model family. / 回到 notebook 调研，或尝试完全不同的模型族。
