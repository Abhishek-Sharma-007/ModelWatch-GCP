# Security Checklist

This document is intentionally a checklist – fast to read, easy to enforce in code review.

## Data

- [x] **Synthetic data only.** The shipped `data/raw/customer_churn_sample.csv` is generated programmatically and contains no real PII.
- [x] No real customer identifiers in the repo. Synthetic `customer_id` strings are not features and are not logged by the API.
- [x] Prediction logs are local CSV by default. `data/logs/*` is `.gitignore`d. **Never commit production prediction logs.**

## Secrets

- [x] No secrets in code or configuration files.
- [x] `.env` is **not** committed. Only `.env.example` is tracked.
- [x] `.env` is in `.gitignore`.
- [x] All configurable values default to safe local values – the project never *requires* a secret to run.
- [x] In production on GCP, secrets live in **Secret Manager** and are mounted into the Cloud Run service as environment variables.

## API

- [x] Pydantic validation on every input. Unknown enum values are rejected with 422.
- [x] Out-of-range numeric inputs (negative tenure, monthly charges > 1000, etc.) are rejected with 422.
- [x] A global exception handler ensures the API never leaks tracebacks to the client.
- [x] Logging mistakes never crash inference - logging is wrapped in `try/except` so a write failure cannot fail a prediction.
- [ ] Auth is **not** included in the demo (no risk locally). For production, put the service behind IAP or a signed API gateway and add `Authorization` header validation.

## Logging hygiene

- [x] Logs include the model input features only (which the client sent us), the prediction, the probability, the model version and the timestamp.
- [x] Logs do not include any authentication header, IP address, or session token.
- [x] Logs are written via a single helper (`src.logger.log_prediction`) so the schema cannot drift.

## Dependencies

- [x] `requirements.txt` pins lower bounds; CI runs on a clean Python 3.10 and 3.11.
- [x] No heavy or unused dependencies (e.g. `seaborn` is deliberately not used).
- [ ] (Recommended) Run `pip-audit` or Dependabot to receive vulnerability notifications.

## GitHub safety checklist

Before `git push`:

- [ ] `git status` shows no `.env`, no `*.joblib` larger than a few MB, no `data/logs/*.csv`.
- [ ] `git diff` contains no API keys, no real names, no real customer rows.
- [ ] All new code has docstrings and type hints.
- [ ] `pytest -v` passes locally.
- [ ] CI (`tests.yml`) is green on the PR.

## Production hardening (out-of-scope for the demo, on the roadmap)

- Add request authentication on `/predict`.
- Rate-limit `/predict` at the load balancer.
- Move logs to BigQuery + a private dataset with column-level access control.
- Encrypt model artifacts at rest (CMEK on the Cloud Storage bucket).
- Add SLO + error budget tracking in Cloud Monitoring.
