Split: `domain` | Train: 10,094 | Validation: 3,298 | Test: 3,250 URLs

Model and thresholds picked on validation, test scores use each model's threshold.

| Model | Val recall @ 1% FPR | Accuracy | Precision | Recall | F1 | False positive rate | ROC AUC | Recall @ 1% FPR |
|---|---|---|---|---|---|---|---|---|
| Baseline (always guess majority) | 0.000 | 0.474 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| Logistic Regression | 0.554 | 0.759 | 0.972 | 0.557 | 0.708 | 0.018 | 0.914 | 0.490 |
| Random Forest | 0.708 | 0.829 | 0.984 | 0.685 | 0.808 | 0.012 | 0.947 | 0.675 |
| XGBoost | 0.710 | 0.836 | 0.981 | 0.701 | 0.818 | 0.015 | 0.949 | 0.570 |
| **Char n-grams** (selected) | 0.710 | 0.848 | 0.982 | 0.724 | 0.833 | 0.015 | 0.970 | 0.669 |
| Char n-grams + features | 0.697 | 0.840 | 0.980 | 0.710 | 0.823 | 0.016 | 0.963 | 0.684 |
