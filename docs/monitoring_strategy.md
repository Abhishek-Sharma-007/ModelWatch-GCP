# Monitoring Strategy

What we monitor, how we measure it, and what triggers human or automated action.

## Layers of monitoring

| Layer                | What it answers                                  | Signal source                       |
|----------------------|--------------------------------------------------|-------------------------------------|
| Service health       | "Is the API up and responsive?"                  | Cloud Run uptime, `/health` probe   |
| API performance      | "Is the API fast and reliable?"                  | Cloud Monitoring (latency, errors)  |
| Input/data drift     | "Are incoming features distributed like before?" | `src.drift_detection`               |
| Prediction drift     | "Are the model's outputs distributed like before?" | Histogram of `churn_probability`   |
| Model quality        | "Do predictions still match ground truth?"       | Labeled feedback in BigQuery        |
| Cost                 | "Are we within budget?"                          | Billing exports + Cloud Monitoring  |

## Model metrics

- Tracked at training time: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Persisted to `reports/metrics.json` and surfaced on the **Model Overview** dashboard page.
- In GCP, these become Cloud Monitoring metrics keyed by `model_version`.

## Data drift

- Numeric drift: normalized absolute mean shift vs reference standard deviation.
- Categorical drift: total variation distance.
- Threshold: configurable via `DRIFT_THRESHOLD` (default 0.20). `low < threshold ≤ medium < 2x threshold ≤ high`.
- Report regenerated whenever the dashboard's **Drift Monitoring** page is opened, or on a Cloud Scheduler cadence in production.

## Prediction drift

- Compare the distribution of `churn_probability` over the latest window vs a reference window.
- Use a Kolmogorov–Smirnov statistic on the probability column.
- A large shift can indicate either real changes in customer mix **or** model degradation; pair with data drift to disambiguate.

## API latency

- Cloud Run reports p50/p95/p99 latency out of the box.
- Alert: p95 > 500ms for 10 minutes.

## Error rate

- Count of 5xx + 4xx (excluding 422 validation errors).
- Alert: error rate > 5% for 5 minutes.
- Validation errors (422) are tracked separately because they are usually client bugs, not service problems.

## Retraining trigger

A retraining run starts when any of the following is true (consume from a single decision module so the policy is auditable):

- Overall drift status is `high` for two consecutive scheduled checks.
- Test-set ROC-AUC on the latest 7 days of labeled traffic drops by more than 0.05 vs the registered model's baseline.
- 30 days have passed since the last successful training run.

The retraining workflow is the same Python code as `run_training.py`. On GCP this is a Vertex AI Pipeline that ends by registering the new model version in Vertex AI Model Registry.

## Alerting recipients

- **PagerDuty / on-call**: service-down, error-rate, latency alerts.
- **Slack channel**: drift `medium` warnings, model-quality regressions.
- **Email**: weekly summary report (overall drift, request count, top features by drift score).
