# MLOps Workflow

End-to-end lifecycle of a model in ModelWatch-GCP, from raw data to retraining.

## 1. Data ingestion

- **Local mode**: `data/raw/customer_churn_sample.csv` is checked in (synthetic data).
- **GCP mode**: raw data lives in Cloud Storage. A scheduled job copies new data into the training bucket.
- Schema validation happens in `src/data_loader.load_raw_data` - missing columns trigger a `ValueError` early.

## 2. Training

- `src.train_model.train_and_select` reads the data, splits it (stratified), fits a `ColumnTransformer + LogisticRegression` and a `ColumnTransformer + RandomForest`, computes metrics, and returns the better-scoring pipeline.
- Random seed is fixed via `RANDOM_STATE` for reproducibility.
- Training is deterministic enough to be CI-friendly.

## 3. Evaluation

- Test-set metrics: accuracy, precision, recall, F1, ROC-AUC, confusion matrix.
- Both candidate models are evaluated; the selected one is highlighted in the markdown report.
- Metrics are persisted in machine-readable JSON (`reports/metrics.json`) and human-readable markdown (`reports/model_report.md`).
- The model card (`reports/model_card.md`) describes intended use, limitations and ethical considerations.

## 4. Model artifact

- The full `sklearn.pipeline.Pipeline` (preprocessing + classifier) is serialized to `models/churn_model.joblib`.
- The artifact embeds preprocessing, so no train/serve skew is possible.
- In GCP, the artifact is copied to Cloud Storage and registered with Vertex AI Model Registry, tagged with `MODEL_VERSION`.

## 5. Deployment

- **Local**: `uvicorn api.fastapi_app:app --reload`.
- **Docker**: build with the provided Dockerfile; run on port 8000.
- **GCP**: `gcloud run deploy modelwatch-api --image <artifact-registry-tag>` with the appropriate service account and secrets.

## 6. Prediction logging

- Every `POST /predict` request writes one row to `data/logs/prediction_logs.csv` (local) or `modelwatch.prediction_logs` BigQuery table (GCP).
- Logs include all input features, the prediction, the probability, the model version and a timestamp. They never contain secrets or auth tokens.

## 7. Drift detection

- Runs on demand (Streamlit dashboard) or on a schedule (Cloud Scheduler + Cloud Run Job).
- Numeric: normalized absolute mean shift vs the reference standard deviation.
- Categorical: total variation distance between category frequencies.
- Per-feature `low/medium/high` status and an overall worst-of status.
- Output: `reports/drift_report.md` + a structured list returned to whoever called the detector.

## 8. Retraining trigger

- A new training run is launched when **any** of:
  - Overall drift status is `high` for two consecutive scheduled checks.
  - Test-set ROC-AUC on the latest 7 days of labeled traffic drops more than 0.05 from the registered model's baseline.
  - A fixed cadence (e.g. monthly) elapses since the last successful training run.
- A retraining run is just `run_training.py` invoked inside a Vertex AI Pipeline / Cloud Run Job. The newly trained model is registered with a bumped `MODEL_VERSION` and rolled out to Cloud Run gradually (revision splitting).

## 9. Continuous integration

- GitHub Actions installs deps and runs `pytest -v` on every push and PR.
- Tests cover data loading, preprocessing, training-on-sample, prediction, drift, and the API (via `TestClient`).
- No cloud credentials are required to run the suite.
