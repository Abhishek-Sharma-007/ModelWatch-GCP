"""ModelWatch-GCP Streamlit monitoring dashboard.

Run with::

    streamlit run ui/streamlit_dashboard.py

The dashboard is intentionally read-only with respect to training: it
visualizes artifacts produced by ``run_training.py`` and the API. The
"Make Prediction" page calls :func:`src.predict.predict_one` directly
so the dashboard still works without the FastAPI service running.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import streamlit as st

# Make ``import src.*`` work when launched via ``streamlit run``.
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

from src.config import SETTINGS  # noqa: E402
from src.logger import log_prediction  # noqa: E402
from src.monitoring import (  # noqa: E402
    compute_drift_dashboard_data,
    load_metrics,
    load_model_card,
    load_recent_predictions,
    load_reference_data,
    summarize_predictions,
)
from src.predict import (  # noqa: E402
    ModelNotLoadedError,
    is_model_available,
    predict_one,
)


st.set_page_config(
    page_title="ModelWatch-GCP",
    page_icon="📊",
    layout="wide",
)


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------

st.sidebar.title("ModelWatch-GCP")
st.sidebar.markdown(
    "Production-style MLOps monitoring for a customer churn model. "
    "All data shown is synthetic."
)
st.sidebar.markdown(f"**Model version:** `{SETTINGS.model_version}`")
st.sidebar.markdown("**Run mode:** local demo")

PAGES = [
    "Model Overview",
    "Make Prediction",
    "Prediction Logs",
    "Drift Monitoring",
    "GCP Architecture",
]
page = st.sidebar.radio("Navigation", PAGES)

model_ok = is_model_available()
st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Model artifact:** {'✅ loaded' if model_ok else '❌ missing'}"
)
if not model_ok:
    st.sidebar.warning("Run `python run_training.py` to generate the model.")


# ---------------------------------------------------------------------------
# Page 1: Model Overview
# ---------------------------------------------------------------------------

def render_model_overview() -> None:
    st.title("📊 Model Overview")
    st.caption("Training metrics, confusion matrix, features and model card.")

    metrics_payload = load_metrics()
    if metrics_payload is None:
        st.warning(
            "No metrics found. Run `python run_training.py` to generate "
            "`reports/metrics.json`."
        )
        return

    st.subheader("Selected model")
    best = metrics_payload.get("best_model", "?")
    st.success(f"**{best}** (model version {metrics_payload.get('model_version')})")

    all_metrics = metrics_payload.get("models", {})
    rows = []
    for name, m in all_metrics.items():
        rows.append(
            {
                "model": name,
                "accuracy": m["accuracy"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "roc_auc": m["roc_auc"],
            }
        )
    metrics_df = pd.DataFrame(rows).set_index("model")
    st.subheader("Metrics")
    st.dataframe(metrics_df.style.format("{:.4f}"))

    st.subheader("Confusion matrix (selected model)")
    cm = all_metrics.get(best, {}).get("confusion_matrix")
    if cm:
        cm_df = pd.DataFrame(
            cm,
            index=["Actual: No Churn", "Actual: Churn"],
            columns=["Pred: No Churn", "Pred: Churn"],
        )
        st.dataframe(cm_df)

    st.subheader("Features used")
    col1, col2 = st.columns(2)
    with col1:
        st.markdown("**Numeric features**")
        st.write(SETTINGS.numeric_features)
    with col2:
        st.markdown("**Categorical features**")
        st.write(SETTINGS.categorical_features)

    st.subheader("Model card")
    card = load_model_card()
    if card:
        st.markdown(card)
    else:
        st.info("Model card not found at `reports/model_card.md`.")


# ---------------------------------------------------------------------------
# Page 2: Make Prediction
# ---------------------------------------------------------------------------

def render_make_prediction() -> None:
    st.title("🔮 Make a Prediction")
    st.caption(
        "Fill out the customer profile and run the churn model. "
        "Predictions made here are logged to "
        "`data/logs/prediction_logs.csv`."
    )

    with st.form("predict_form"):
        col1, col2 = st.columns(2)
        with col1:
            tenure_months = st.number_input(
                "Tenure (months)", min_value=0, max_value=120, value=12
            )
            monthly_charges = st.number_input(
                "Monthly charges ($)",
                min_value=0.0,
                max_value=1000.0,
                value=75.5,
                step=0.5,
            )
            total_charges = st.number_input(
                "Total charges ($)",
                min_value=0.0,
                max_value=100000.0,
                value=906.0,
                step=10.0,
            )
            support_tickets = st.number_input(
                "Support tickets", min_value=0, max_value=100, value=3
            )
            is_senior_citizen = st.selectbox(
                "Is senior citizen?", options=[0, 1], index=0
            )
        with col2:
            contract_type = st.selectbox(
                "Contract type",
                ["Month-to-month", "One year", "Two year"],
            )
            internet_service = st.selectbox(
                "Internet service", ["DSL", "Fiber optic", "No"]
            )
            payment_method = st.selectbox(
                "Payment method",
                [
                    "Electronic check",
                    "Mailed check",
                    "Bank transfer",
                    "Credit card",
                ],
            )

        submitted = st.form_submit_button("Predict")

    if not submitted:
        return

    payload = {
        "tenure_months": int(tenure_months),
        "monthly_charges": float(monthly_charges),
        "total_charges": float(total_charges),
        "contract_type": contract_type,
        "internet_service": internet_service,
        "support_tickets": int(support_tickets),
        "payment_method": payment_method,
        "is_senior_citizen": int(is_senior_citizen),
    }

    try:
        result = predict_one(payload)
    except ModelNotLoadedError as exc:
        st.error(str(exc))
        return
    except Exception as exc:  # pragma: no cover - defensive UI
        st.error(f"Prediction failed: {exc}")
        return

    try:
        log_prediction(payload, result)
    except Exception:  # pragma: no cover
        st.warning("Prediction succeeded but logging failed.")

    label = result["prediction_label"]
    prob = result["churn_probability"]
    if result["prediction"] == 1:
        st.error(f"Prediction: **{label}** (probability {prob:.2%})")
    else:
        st.success(f"Prediction: **{label}** (churn probability {prob:.2%})")
    st.progress(min(max(prob, 0.0), 1.0))
    with st.expander("Raw response"):
        st.json(result)


# ---------------------------------------------------------------------------
# Page 3: Prediction Logs
# ---------------------------------------------------------------------------

def render_prediction_logs() -> None:
    st.title("🧾 Prediction Logs")
    st.caption("Recent predictions served by the API and the dashboard.")

    df = load_recent_predictions()
    summary = summarize_predictions(df)

    cols = st.columns(4)
    cols[0].metric("Total predictions", summary["total_predictions"])
    cols[1].metric("Predicted churn", summary["churn_count"])
    cols[2].metric("Predicted no-churn", summary["no_churn_count"])
    cols[3].metric("Avg churn prob.", f"{summary['avg_probability']:.2%}")

    if df.empty:
        st.info(
            "No predictions logged yet. Use the **Make Prediction** page "
            "or call the FastAPI `/predict` endpoint to generate logs."
        )
        return

    st.subheader("Latest predictions (most recent 50)")
    st.dataframe(df.tail(50).iloc[::-1])

    st.subheader("Churn probability distribution")
    fig, ax = plt.subplots(figsize=(6, 3))
    ax.hist(df["churn_probability"].astype(float), bins=20, edgecolor="black")
    ax.set_xlabel("Churn probability")
    ax.set_ylabel("Count")
    ax.set_xlim(0, 1)
    st.pyplot(fig)

    st.subheader("Predicted label counts")
    label_counts = df["prediction_label"].value_counts()
    st.bar_chart(label_counts)


# ---------------------------------------------------------------------------
# Page 4: Drift Monitoring
# ---------------------------------------------------------------------------

def _status_color(status: str) -> str:
    return {"low": "🟢", "medium": "🟡", "high": "🔴"}.get(status, "⚪")


def render_drift_monitoring() -> None:
    st.title("📉 Drift Monitoring")
    st.caption(
        "Compares reference training data to recent prediction logs. "
        "Drift report is written to `reports/drift_report.md`."
    )

    drift = compute_drift_dashboard_data()
    overall = drift["overall"]
    st.markdown(
        f"### Overall drift status: {_status_color(overall)} **{overall.upper()}**"
    )
    st.write(
        f"Drift threshold (low → medium boundary): **{drift['threshold']:.2f}**. "
        f"Predictions analyzed: **{drift['n_logs']}**."
    )

    if not drift["has_logs"]:
        st.info(
            "No prediction logs available yet, so drift cannot be computed. "
            "Use the **Make Prediction** page first."
        )
        return

    rows = [
        {
            "feature": r["feature"],
            "type": r["type"],
            "drift_score": r["drift_score"],
            "status": f"{_status_color(r['status'])} {r['status']}",
        }
        for r in drift["results"]
    ]
    st.dataframe(pd.DataFrame(rows))

    st.subheader("Feature distribution comparison")
    ref_df = load_reference_data()
    cur_df = load_recent_predictions()

    feature = st.selectbox(
        "Feature to inspect",
        SETTINGS.numeric_features + SETTINGS.categorical_features,
    )

    if feature in SETTINGS.numeric_features:
        fig, ax = plt.subplots(figsize=(6, 3))
        ax.hist(
            ref_df[feature].dropna().astype(float),
            bins=20,
            alpha=0.6,
            label="reference",
            density=True,
            edgecolor="black",
        )
        if feature in cur_df.columns and len(cur_df) > 0:
            ax.hist(
                cur_df[feature].dropna().astype(float),
                bins=20,
                alpha=0.6,
                label="current",
                density=True,
                edgecolor="black",
            )
        ax.set_xlabel(feature)
        ax.set_ylabel("density")
        ax.legend()
        st.pyplot(fig)
    else:
        ref_freq = (
            ref_df[feature].astype(str).value_counts(normalize=True).rename("reference")
        )
        cur_freq = (
            cur_df[feature].astype(str).value_counts(normalize=True).rename("current")
            if feature in cur_df.columns
            else pd.Series(dtype=float, name="current")
        )
        freq_df = pd.concat([ref_freq, cur_freq], axis=1).fillna(0.0)
        st.bar_chart(freq_df)

    st.subheader("Drift report (markdown)")
    report_path = SETTINGS.reports_dir / "drift_report.md"
    if report_path.exists():
        st.markdown(report_path.read_text())


# ---------------------------------------------------------------------------
# Page 5: GCP Architecture
# ---------------------------------------------------------------------------

def render_gcp_architecture() -> None:
    st.title("☁️ GCP Architecture")
    st.caption(
        "How ModelWatch-GCP maps onto Google Cloud services for a "
        "production deployment."
    )

    components = [
        ("Cloud Run", "Hosts the FastAPI prediction service as a stateless container."),
        ("Vertex AI", "Trains models, registers model versions and (optionally) hosts an endpoint with built-in monitoring."),
        ("Cloud Storage", "Stores raw datasets, processed training snapshots and serialized model artifacts."),
        ("BigQuery", "Production-grade store for prediction logs - replaces the local CSV file."),
        ("Cloud Logging", "Captures structured logs from Cloud Run (stdout/stderr) for observability."),
        ("Cloud Monitoring", "Tracks API latency, error rate and custom drift metrics with alert policies."),
        ("Secret Manager", "Stores any secrets (DB credentials, third-party API keys) so they never live in code or env files."),
        ("Cloud Scheduler", "Triggers periodic drift evaluation and retraining jobs."),
    ]
    for name, desc in components:
        st.markdown(f"**{name}** - {desc}")

    st.subheader("High-level flow")
    st.markdown(
        """
1. **Training** runs on Vertex AI (or Cloud Run jobs). Outputs are written to Cloud Storage and registered in Vertex AI Model Registry.
2. **Serving** is a Cloud Run service running this FastAPI app. Cloud Run pulls the model artifact at startup or from Vertex AI.
3. **Logging** writes every prediction to BigQuery via streaming inserts (instead of CSV in local mode).
4. **Monitoring** uses Cloud Monitoring dashboards for latency / errors and a Cloud Scheduler job that runs drift detection daily.
5. **Retraining** is triggered when drift status crosses a threshold or on a fixed cadence.
        """
    )

    plan_path = _PROJECT_ROOT / "docs" / "gcp_deployment_plan.md"
    if plan_path.exists():
        with st.expander("Full GCP deployment plan"):
            st.markdown(plan_path.read_text())


# ---------------------------------------------------------------------------
# Router
# ---------------------------------------------------------------------------

if page == "Model Overview":
    render_model_overview()
elif page == "Make Prediction":
    render_make_prediction()
elif page == "Prediction Logs":
    render_prediction_logs()
elif page == "Drift Monitoring":
    render_drift_monitoring()
elif page == "GCP Architecture":
    render_gcp_architecture()
