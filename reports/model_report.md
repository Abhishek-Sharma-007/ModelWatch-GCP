# Model Report

Model version: **v1.0.0**

**Selected model:** `logistic_regression`

## Model: logistic_regression

| Metric | Value |
|--------|-------|
| Accuracy  | 0.7667 |
| Precision | 0.5478 |
| Recall    | 0.7778 |
| F1-score  | 0.6429 |
| ROC-AUC   | 0.8401 |

### Confusion matrix

| | Pred: No Churn | Pred: Churn |
|---|---|---|
| **Actual: No Churn** | 167 | 52 |
| **Actual: Churn**    | 18 | 63 |

## Model: random_forest

| Metric | Value |
|--------|-------|
| Accuracy  | 0.7800 |
| Precision | 0.5882 |
| Recall    | 0.6173 |
| F1-score  | 0.6024 |
| ROC-AUC   | 0.8352 |

### Confusion matrix

| | Pred: No Churn | Pred: Churn |
|---|---|---|
| **Actual: No Churn** | 184 | 35 |
| **Actual: Churn**    | 31 | 50 |
