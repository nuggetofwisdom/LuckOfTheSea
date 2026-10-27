Split: `domain` | Train: 8,712 URLs | Test: 2,140 URLs

| Model | Accuracy | Precision | Recall | F1 | False positive rate | ROC AUC | Recall @ 1% FPR |
|---|---|---|---|---|---|---|---|
| Baseline (always guess majority) | 0.495 | 0.495 | 1.000 | 0.662 | 1.000 | 0.500 | 0.000 |
| Logistic Regression | 0.852 | 0.882 | 0.808 | 0.844 | 0.105 | 0.943 | 0.656 |
| Random Forest | 0.904 | 0.914 | 0.889 | 0.901 | 0.081 | 0.965 | 0.690 |
| XGBoost | 0.898 | 0.893 | 0.903 | 0.898 | 0.106 | 0.964 | 0.687 |
