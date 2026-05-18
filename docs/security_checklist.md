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
- [x] Security headers are attached to every API response.
- [x] CORS is disabled by default and can be enabled only with an explicit allow-list.
- [x] `/predict` supports optional `X-API-Key` validation when `API_KEY` is supplied by the runtime environment.
- [x] Logging mistakes never crash inference - logging is wrapped in `try/except` so a write failure cannot fail a prediction.

## Logging hygiene

- [x] Logs include the model input features only (which the client sent us), the prediction, the probability, the model version and the timestamp.
- [x] Logs do not include any authentication header, IP address, or session token.
- [x] Logs are written via a single helper (`src.logger.log_prediction`) so the schema cannot drift.

## Dependencies

- [x] `requirements.txt` pins lower bounds; CI runs on a clean Python 3.10 and 3.11.
- [x] No heavy or unused dependencies (e.g. `seaborn` is deliberately not used).
- [x] CI runs the dependency-free `python scripts/security_audit.py` scanner for common committed-secret patterns.
- [ ] (Recommended) Run `pip-audit` or Dependabot to receive vulnerability notifications.

## GitHub safety checklist

Before `git push`:

- [ ] `git status` shows no `.env`, no `*.joblib` larger than a few MB, no `data/logs/*.csv`.
- [ ] `git diff` contains no API keys, no real names, no real customer rows.
- [ ] All new code has docstrings and type hints.
- [ ] `python scripts/security_audit.py` reports no obvious secrets.
- [ ] `pytest -v` passes locally.
- [ ] CI (`tests.yml`) is green on the PR.

## Production hardening (out-of-scope for the demo, on the roadmap)

- For internet-facing deployments, keep `API_KEY` enabled and also place the service behind IAM/IAP or an API gateway.
- Rate-limit `/predict` at the load balancer.
- Move logs to BigQuery + a private dataset with column-level access control.
- Encrypt model artifacts at rest (CMEK on the Cloud Storage bucket).
- Add SLO + error budget tracking in Cloud Monitoring.
