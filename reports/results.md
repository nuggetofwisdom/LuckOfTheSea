Split: `domain` | Train: 10,094 | Validation: 3,298 | Test: 3,250 URLs

Model and thresholds picked on validation, test scores use each model's threshold.

| Model | Val avg recall @ 0-5% FPR | Accuracy | Precision | Recall | F1 | False positive rate | ROC AUC | Recall @ 1% FPR |
|---|---|---|---|---|---|---|---|---|
| Baseline (always guess majority) | 0.025 | 0.474 | 0.000 | 0.000 | 0.000 | 0.000 | 0.500 | 0.000 |
| Logistic Regression | 0.592 | 0.760 | 0.984 | 0.553 | 0.708 | 0.010 | 0.918 | 0.577 |
| Random Forest | 0.735 | 0.824 | 0.985 | 0.675 | 0.801 | 0.011 | 0.947 | 0.675 |
| XGBoost | 0.742 | 0.825 | 0.981 | 0.680 | 0.803 | 0.015 | 0.948 | 0.593 |
| Char n-grams | 0.750 | 0.848 | 0.982 | 0.724 | 0.833 | 0.015 | 0.970 | 0.669 |
| Char n-grams + features | 0.739 | 0.850 | 0.980 | 0.730 | 0.836 | 0.017 | 0.965 | 0.693 |
| **XGBoost + Char n-grams (averaged)** (selected) | 0.841 | 0.881 | 0.982 | 0.788 | 0.874 | 0.016 | 0.974 | 0.756 |
