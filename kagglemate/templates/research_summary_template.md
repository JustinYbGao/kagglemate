# Research Summary — {{ competition_name }}
>
> **Competition**: {{ competition_slug }}
> **Generated**: {{ generated_at }}
> **Notebooks analyzed**: {{ notebook_count }}

---

## Top Public Approaches

{% for nb in notebooks %}
### {{ loop.index }}. {{ nb.title }}
- **Author**: {{ nb.author }}
- **Votes**: {{ nb.votes }}
- **Model**: {{ nb.model }}
{% if nb.lb_score %}- **LB Score**: {{ nb.lb_score }}{% endif %}
- **CV Method**: {{ nb.cv_method }}
- **Key Techniques**: {{ nb.key_techniques | join(", ") }}
- **Worth Reproducing**: {{ "Yes" if nb.worth_reproducing else "No" }}
- **Notes**: {{ nb.notes }}

{% endfor %}

---

## Common Patterns Across Top Notebooks

{{ common_patterns }}

---

## Potential Baseline

Based on the most common successful approach:

- **Model**: {{ recommended_model }}
- **CV**: {{ recommended_cv }}
- **Feature engineering**: {{ recommended_fe }}

---

## High-ROI Improvement Ideas

{% for idea in improvement_ideas %}
{{ loop.index }}. {{ idea }}
{% endfor %}

---

## Notebooks to Study in Detail

{% for nb in notebooks_to_study %}- [**{{ nb.ref }}**](https://www.kaggle.com/code/{{ nb.ref }}) — {{ nb.reason }}
{% endfor %}
