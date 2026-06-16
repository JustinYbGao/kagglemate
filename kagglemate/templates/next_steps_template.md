# Next Steps — {{ competition_name }}
>
> **Competition**: {{ competition_slug }}
> **Generated**: {{ generated_at }}
> **Total experiments**: {{ total_experiments }}

---

## Current Best

| Metric | Value |
|--------|-------|
| CV Score | {{ best_cv }} |
| LB Score | {{ best_lb }} |

---

## Situation Diagnosis

{{ situation }}

---

## Recommended Next Experiments

{% for rec in recommendations %}
### {{ loop.index }}. {{ rec.name | replace("_", " ") | title }}
- **Impact**: {{ rec.expected_impact | upper }}
- **Estimated CV improvement**: {{ rec.cv_improvement_estimate }}
- **What to do**: {{ rec.what_to_do }}
- **Why**: {{ rec.reason }}
- **Risk**: {{ rec.risk }}

{% endfor %}

---

## Decision

Choose ONE experiment from the recommendations above. To implement:

```bash
# Edit the baseline script with the chosen technique,
# or run the agent to generate a new experiment:
python main.py baseline --competition {{ competition_slug }}

# Then run it:
python main.py run --competition {{ competition_slug }}

# Then get new suggestions:
python main.py suggest --competition {{ competition_slug }}
```

---

## General Guidance

1. **Don't chase LB directly** — optimize CV, not public leaderboard.
2. **One change at a time** — change one thing per experiment so you know what worked.
3. **CV/LB gap is your compass** — wide gap → reduce complexity. Tight gap → add complexity.
4. **Feature engineering > Hyperparameter tuning** — new features usually beat better params.
5. **When stuck**: go back to notebooks, or try a completely different model family.
