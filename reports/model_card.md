# Model Card – ModelWatch-GCP churn model

## Model purpose
Predict the probability that a subscription customer will churn (cancel their contract) in the near term, so that downstream business systems (retention campaigns, support prioritization, finance forecasts) can act on the prediction.

## Model type
A `sklearn.pipeline.Pipeline` consisting of:
- `ColumnTransformer` (median imputation + standard scaling for numeric, most-frequent imputation + one-hot encoding for categorical),
- a binary classifier - either `LogisticRegression(class_weight="balanced", max_iter=1000)` or `RandomForestClassifier(n_estimators=200, max_depth=10, class_weight="balanced")`.

At training time both candidates are fit and the one with the higher held-out **ROC-AUC** is persisted.

## Dataset
- Source: synthetic, generated programmatically and shipped in this repo (`data/raw/customer_churn_sample.csv`).
- Size: 1,500 rows, ~27% churn rate.
- No real personal data. Identifiers (`customer_id`) are synthetic and are not used as features.

## Features
- Numeric: `tenure_months`, `monthly_charges`, `total_charges`, `support_tickets`, `is_senior_citizen`.
- Categorical: `contract_type`, `internet_service`, `payment_method`.
- Target: `churn` (0/1).

## Training procedure
- Stratified 80/20 train/test split with `random_state=42`.
- The whole pipeline is trained on the train split.
- Metrics reported on the held-out test split.

## Evaluation
- Metrics tracked: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Persisted to `reports/metrics.json` and `reports/model_report.md` after each training run.

## Intended use
- Educational / portfolio demonstration of an MLOps stack.
- Reference architecture for a churn-prediction service on Google Cloud.

## Not intended for
- Real customer-facing decisions. The dataset is synthetic; absolute metric values are not generalizable.
- High-stakes use (credit decisions, eligibility, healthcare, etc.). Use a properly validated, fairness-audited model for those.

## Limitations
- Synthetic data means feature distributions are by construction; the model may behave very differently on real data.
- The model is a simple tabular classifier; it does not encode any time-series structure (e.g. month-over-month behavioral changes).
- No protected-attribute analysis is performed because the synthetic data does not contain protected attributes; do not assume the model is fairness-audited.

## Ethical considerations
- Customer churn predictions can drive retention offers or service deprioritization. Any deployment should ensure that interventions do not discriminate against protected groups.
- The `is_senior_citizen` feature is a proxy for age and would need a fairness audit before any production use.
- The pipeline is deliberately interpretable (Logistic Regression baseline is shipped alongside Random Forest) so that retention teams can sanity-check why a prediction was made.

## Monitoring plan
- Input data drift is monitored continuously by `src/drift_detection.py` (or Vertex AI Model Monitoring in production).
- Prediction drift (distribution of `churn_probability`) is tracked in the Streamlit dashboard.
- Retraining is triggered when overall drift status reaches `high` for two consecutive scheduled checks, when test-set ROC-AUC on the latest labeled traffic drops by more than 0.05, or on a 30-day cadence.
- Every prediction is logged with the `model_version`, so when the model is updated, old vs new predictions remain distinguishable in analytics.
