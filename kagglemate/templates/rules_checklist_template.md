# Rules Checklist — {{ competition_name }}
>
> **Competition**: {{ competition_slug }}
> **Generated**: {{ generated_at }}
>
> ⚠️ **IMPORTANT**: This checklist is auto-generated. Some rules require manual verification.
> Before submitting, review EVERY item below.

---

## Submission Rules

- [ ] **Submission format**: CSV with columns `{{ submission_cols }}` and {{ submission_rows }} rows
- [ ] **File type**: {{ file_type_note }}
- [ ] **Daily submission limit**: {{ daily_limit_note }}
- [ ] **Max team size**: {{ team_size_note }}

## Code & Data Rules

- [ ] **External data allowed?** {{ external_data_note }}
- [ ] **Internet access in notebooks?** {{ internet_note }}
- [ ] **Pre-trained models allowed?** {{ pretrained_note }}
- [ ] **GPU required?** {{ gpu_note }}

## Competition-Specific Rules

- [ ] **Code Competition?** {{ code_competition_note }}
- [ ] **Late submission allowed?** {{ late_submission_note }}
- [ ] **Must use competition data only?** Verify on competition rules page.

## Submission Checklist (Run Before Every Submit)

```bash
python main.py submission validate --competition {{ competition_slug }} --file <path>
```

- [ ] File exists and is not empty
- [ ] Row count matches test set
- [ ] Column names match sample_submission exactly
- [ ] ID column values match test set
- [ ] No NaN values
- [ ] No infinite values
- [ ] Prediction values in valid range

## Common Pitfalls

1. **Double file extension** — If filename column already includes `.jpg`, don't append it again.
2. **Wrong ID order** — IDs in submission must match test set order, not train set.
3. **Zipped submission** — Some competitions require `.zip`. Check sample submission or competition rules.
4. **Silent failures** — If the training script catches exceptions silently, your submission might be full of zeros/NaNs without you knowing.

## Confirmation

> I have reviewed all items above and confirm this submission complies with competition rules.

- [ ] Signature: _______________
- [ ] Date: {{ generated_at }}
