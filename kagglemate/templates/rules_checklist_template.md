# Rules Checklist / 规则检查清单 — {{ competition_name }}
>
> **Competition / 比赛**: {{ competition_slug }}
> **Generated / 生成时间**: {{ generated_at }}
>
> ⚠️ **IMPORTANT / 重要提示**: This checklist is auto-generated. Some rules require manual verification. / 此清单由 AI 自动生成，部分规则需人工核实。
> Before submitting, review EVERY item below. / 提交前请逐项检查。

---

## Submission Rules / 提交规则

- [ ] **Submission format / 提交格式**: CSV with columns / 包含以下列 `{{ submission_cols }}`，{{ submission_rows }} rows / 行
- [ ] **File type / 文件类型**: {{ file_type_note }}
- [ ] **Daily submission limit / 每日提交次数限制**: {{ daily_limit_note }}
- [ ] **Max team size / 最大队伍人数**: {{ team_size_note }}

## Code & Data Rules / 代码与数据规则

- [ ] **External data allowed? / 是否允许外部数据？** {{ external_data_note }}
- [ ] **Internet access in notebooks? / Notebook 能否联网？** {{ internet_note }}
- [ ] **Pre-trained models allowed? / 是否允许预训练模型？** {{ pretrained_note }}
- [ ] **GPU required? / 是否需要 GPU？** {{ gpu_note }}

## Competition-Specific Rules / 比赛特定规则

- [ ] **Code Competition? / 是否为代码竞赛？** {{ code_competition_note }}
- [ ] **Late submission allowed? / 是否允许延迟提交？** {{ late_submission_note }}
- [ ] **Must use competition data only? / 只能使用比赛数据？** Verify on competition rules page. / 请在比赛规则页面核实。

## Submission Checklist / 提交前检查 (Run Before Every Submit / 每次提交前执行)

```bash
python main.py submission validate --competition {{ competition_slug }} --file <path>
```

- [ ] File exists and is not empty / 文件存在且非空
- [ ] Row count matches test set / 行数与测试集一致
- [ ] Column names match sample_submission exactly / 列名与 sample_submission 完全一致
- [ ] ID column values match test set / ID 列值与测试集一致
- [ ] No NaN values / 无 NaN 值
- [ ] No infinite values / 无无穷值
- [ ] Prediction values in valid range / 预测值在合理区间

## Common Pitfalls / 常见陷阱

1. **Double file extension / 重复文件后缀** — If filename column already includes `.jpg`, don't append it again. / 如果文件名列已含 `.jpg`，不要再追加。
2. **Wrong ID order / ID 顺序错误** — IDs in submission must match test set order, not train set. / 提交文件中的 ID 顺序应与测试集一致，而非训练集。
3. **Zipped submission / 需压缩提交** — Some competitions require `.zip`. Check sample submission or competition rules. / 部分比赛要求 `.zip` 格式，请查看示例提交文件或规则。
4. **Silent failures / 静默失败** — If the training script catches exceptions silently, your submission might be full of zeros/NaNs without you knowing. / 如果训练脚本静默捕获了异常，提交文件可能全是 0 或 NaN。

## Confirmation / 确认

> I have reviewed all items above and confirm this submission complies with competition rules. / 我已审阅以上所有内容，确认本次提交符合比赛规则。

- [ ] Signature / 签名: _______________
- [ ] Date / 日期: {{ generated_at }}
