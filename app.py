from dotenv import load_dotenv
load_dotenv()

import json
import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

import llm_client
from shap_explainer import build_explainer, explain_single, FEATURE_LABELS

DATA_PATH = os.path.join("data", "processed_appointments.csv")
MODEL_PATH = os.path.join("model", "noshow_model.pkl")
PLOTS_DIR = "plots"

DAY_NAMES = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
AGE_GROUP_ORDER = ["child", "teen", "young_adult", "adult", "senior"]

st.set_page_config(page_title="Patient No-Show Prediction", layout="wide")

# Cached resources
@st.cache_resource
def load_model_artifact():
    with open(MODEL_PATH, "rb") as f:
        return pickle.load(f)


@st.cache_resource
def load_shap_explainer(_model):
    return build_explainer(_model)


@st.cache_resource
def load_llm_client():
    return llm_client.get_client()


@st.cache_data
def load_data():
    df = pd.read_csv(DATA_PATH, parse_dates=["scheduled_day", "appointment_day"])
    return df


# Shared precomputed aggregates (used by Tab 1 charts AND Tab 4 AI grounding)
@st.cache_data
def build_dashboard_summary(df: pd.DataFrame) -> dict:
    """
    Precompute every aggregate the dashboard/AI layer narrates, into one
    JSON-serialisable dict. This is the ONLY numeric source of truth the
    LLM is allowed to draw from in Tab 4.
    """
    total_appts = int(len(df))
    total_noshows = int(df["no_show"].sum())
    noshow_rate = float(df["no_show"].mean())
    unique_patients = int(df["patient_id"].nunique())

    by_dow = (
        df.groupby("appt_day_of_week")["no_show"].mean().reindex(range(7)).fillna(0)
    )
    noshow_by_day_of_week = {DAY_NAMES[i]: float(by_dow.loc[i]) for i in range(7) if i in by_dow.index}

    lead_bins = [-1, 0, 3, 7, 14, 30, 10_000]
    lead_labels = ["same_day", "1-3_days", "4-7_days", "8-14_days", "15-30_days", "30+_days"]
    df_lead = df.copy()
    df_lead["lead_bucket"] = pd.cut(df_lead["lead_days"], bins=lead_bins, labels=lead_labels)
    noshow_by_lead_bucket = (
        df_lead.groupby("lead_bucket", observed=True)["no_show"].mean().to_dict()
    )
    noshow_by_lead_bucket = {str(k): float(v) for k, v in noshow_by_lead_bucket.items()}

    noshow_by_age_group = (
        df.groupby("age_group")["no_show"].mean().reindex(AGE_GROUP_ORDER).dropna().to_dict()
    )
    noshow_by_age_group = {str(k): float(v) for k, v in noshow_by_age_group.items()}

    sms_impact = df.groupby("sms_received")["no_show"].mean().to_dict()
    sms_impact = {("sms_received" if k == 1 else "no_sms"): float(v) for k, v in sms_impact.items()}

    top_neighbourhoods = (
        df.groupby("neighbourhood")["no_show"]
        .agg(["mean", "count"])
        .query("count >= 50")
        .sort_values("mean", ascending=False)
        .head(5)["mean"]
        .to_dict()
    )
    top_neighbourhoods = {str(k): float(v) for k, v in top_neighbourhoods.items()}

    return {
        "total_appointments": total_appts,
        "total_no_shows": total_noshows,
        "overall_no_show_rate": round(noshow_rate, 4),
        "unique_patients": unique_patients,
        "no_show_rate_by_day_of_week": {k: round(v, 4) for k, v in noshow_by_day_of_week.items()},
        "no_show_rate_by_lead_time_bucket": {k: round(v, 4) for k, v in noshow_by_lead_bucket.items()},
        "no_show_rate_by_age_group": {k: round(v, 4) for k, v in noshow_by_age_group.items()},
        "no_show_rate_sms_impact": {k: round(v, 4) for k, v in sms_impact.items()},
        "top5_highest_noshow_neighbourhoods": {k: round(v, 4) for k, v in top_neighbourhoods.items()},
    }

# App
st.title("Patient No-Show Prediction - AI Insights Dashboard")
st.caption("Kaggle Medical Appointment No-Shows dataset - XGBoost + SHAP + LLM narration layer")

df = load_data()
summary = build_dashboard_summary(df)
artifact = load_model_artifact()
model = artifact["model"]
scaler = artifact["scaler"]
feature_names = artifact["feature_names"]
metrics = artifact["metrics"]
explainer = load_shap_explainer(model)
hf_client = load_llm_client()

if hf_client is None:
    st.sidebar.warning(
        "HF_TOKEN is not set - AI narration features (Tab 2 explanation, Tab 4 chat & "
        "summary) will show raw data instead of LLM text. Get a free token at "
        "huggingface.co/settings/tokens and set it as the HF_TOKEN environment variable."
    )

tab1, tab2, tab3, tab4 = st.tabs(
    ["Dataset Insights", "Risk Predictor", "Model Performance", "AI Insights"]
)

# TAB 1 - Dataset Insights
with tab1:
    st.subheader("Key metrics")
    k1, k2, k3, k4 = st.columns(4)
    k1.metric("Total appointments", f"{summary['total_appointments']:,}")
    k2.metric("Total no-shows", f"{summary['total_no_shows']:,}")
    k3.metric("No-show rate", f"{summary['overall_no_show_rate']*100:.1f}%")
    k4.metric("Unique patients", f"{summary['unique_patients']:,}")

    st.divider()
    c1, c2 = st.columns(2)

    with c1:
        st.markdown("**No-show rate by day of week**")
        dow_df = pd.DataFrame({
            "day": list(summary["no_show_rate_by_day_of_week"].keys()),
            "rate": list(summary["no_show_rate_by_day_of_week"].values()),
        })
        dow_df["day"] = pd.Categorical(dow_df["day"], categories=DAY_NAMES, ordered=True)
        dow_df = dow_df.sort_values("day")
        st.bar_chart(dow_df.set_index("day"))

        st.markdown("**No-show rate by age group**")
        ag_df = pd.DataFrame({
            "age_group": list(summary["no_show_rate_by_age_group"].keys()),
            "rate": list(summary["no_show_rate_by_age_group"].values()),
        })
        ag_df["age_group"] = pd.Categorical(ag_df["age_group"], categories=AGE_GROUP_ORDER, ordered=True)
        ag_df = ag_df.sort_values("age_group")
        st.bar_chart(ag_df.set_index("age_group"))

    with c2:
        st.markdown("**No-show rate by lead-time bucket**")
        lb_order = ["same_day", "1-3_days", "4-7_days", "8-14_days", "15-30_days", "30+_days"]
        lb_df = pd.DataFrame({
            "bucket": list(summary["no_show_rate_by_lead_time_bucket"].keys()),
            "rate": list(summary["no_show_rate_by_lead_time_bucket"].values()),
        })
        lb_df["bucket"] = pd.Categorical(lb_df["bucket"], categories=lb_order, ordered=True)
        lb_df = lb_df.sort_values("bucket")
        st.bar_chart(lb_df.set_index("bucket"))

        st.markdown("**SMS reminder impact on no-show rate**")
        sms_df = pd.DataFrame({
            "group": ["No SMS", "Received SMS"],
            "rate": [
                summary["no_show_rate_sms_impact"].get("no_sms", 0),
                summary["no_show_rate_sms_impact"].get("sms_received", 0),
            ],
        })
        st.bar_chart(sms_df.set_index("group"))

    st.divider()
    st.subheader("Key insights")
    lb_sorted = sorted(summary["no_show_rate_by_lead_time_bucket"].items(), key=lambda x: -x[1])
    dow_sorted = sorted(summary["no_show_rate_by_day_of_week"].items(), key=lambda x: -x[1])
    st.markdown(f"""
- Overall no-show rate is **{summary['overall_no_show_rate']*100:.1f}%** across
  **{summary['total_appointments']:,}** appointments and **{summary['unique_patients']:,}** unique patients.
- Longer lead times correlate strongly with no-shows - the **`{lb_sorted[0][0]}`** bucket has the
  highest no-show rate at **{lb_sorted[0][1]*100:.1f}%**.
- **{dow_sorted[0][0]}** has the highest no-show rate by day of week, at **{dow_sorted[0][1]*100:.1f}%**.
- SMS reminders show a rate of **{summary['no_show_rate_sms_impact'].get('sms_received', 0)*100:.1f}%**
  (received) vs **{summary['no_show_rate_sms_impact'].get('no_sms', 0)*100:.1f}%** (not received) -
  note this is a correlation, not a causal SMS effect (reminders are often sent for appointments
  booked further in advance, which independently raises no-show risk).
""")

# TAB 2 - Risk Predictor
with tab2:
    st.subheader("Predict no-show risk for a patient")

    with st.form("predict_form"):
        c1, c2, c3 = st.columns(3)
        with c1:
            age = st.number_input("Age", min_value=0, max_value=115, value=35)
            gender = st.selectbox("Gender", ["F", "M"])
            lead_days = st.number_input("Lead time (days between booking and appointment)",
                                         min_value=0, max_value=365, value=7)
            sms_received = st.selectbox("SMS reminder received?", ["No", "Yes"]) == "Yes"
        with c2:
            appt_day_of_week = st.selectbox("Appointment day of week", DAY_NAMES)
            scheduled_hour = st.slider("Hour appointment was booked", 0, 23, 10)
            patient_appt_count = st.number_input("Patient's total appointment count", min_value=1,
                                                   max_value=200, value=1)
            neighbourhood_noshowrate = st.slider("Neighbourhood historical no-show rate", 0.0, 1.0,
                                                   float(summary["overall_no_show_rate"]))
        with c3:
            hypertension = st.checkbox("Hypertension")
            diabetes = st.checkbox("Diabetes")
            alcoholism = st.checkbox("Alcoholism")
            handicap = st.checkbox("Handicap")
            scholarship = st.checkbox("Welfare scholarship")

        submitted = st.form_submit_button("Predict risk")

    if submitted:
        dow_idx = DAY_NAMES.index(appt_day_of_week)
        if age <= 12:
            age_group_encoded = 0
        elif age <= 19:
            age_group_encoded = 1
        elif age <= 35:
            age_group_encoded = 2
        elif age <= 60:
            age_group_encoded = 3
        else:
            age_group_encoded = 4

        comorbidity_score = int(hypertension) + int(diabetes) + int(alcoholism) + int(handicap)

        row = pd.DataFrame([{
            "age": age,
            "scholarship": int(scholarship),
            "hypertension": int(hypertension),
            "diabetes": int(diabetes),
            "alcoholism": int(alcoholism),
            "handicap": int(handicap),
            "sms_received": int(sms_received),
            "lead_days": lead_days,
            "appt_day_of_week": dow_idx,
            "appt_day_of_month": 15,
            "scheduled_hour": scheduled_hour,
            "is_weekend": int(dow_idx in [5, 6]),
            "comorbidity_score": comorbidity_score,
            "patient_appt_count": patient_appt_count,
            "neighbourhood_noshowrate": neighbourhood_noshowrate,
            "gender_encoded": 0 if gender == "F" else 1,
            "age_group_encoded": age_group_encoded,
        }])[feature_names]

        row_scaled = scaler.transform(row)
        risk = float(model.predict_proba(row_scaled)[0, 1])
        st.session_state["last_prediction"] = {
            "risk_pct": risk * 100,
            "row": row,
            "row_scaled": row_scaled,
        }

    if "last_prediction" in st.session_state:
        pred = st.session_state["last_prediction"]
        risk_pct = pred["risk_pct"]

        if risk_pct < 25:
            band, color, rec = "LOW", "green", "Standard scheduling - no extra action needed."
        elif risk_pct < 55:
            band, color, rec = "MODERATE", "orange", "Consider sending an SMS/call reminder 24-48h before the appointment."
        else:
            band, color, rec = "HIGH", "red", "Recommend a phone call confirmation and/or overbooking this slot."

        st.divider()
        m1, m2 = st.columns([1, 2])
        with m1:
            st.metric("Predicted no-show risk", f"{risk_pct:.1f}%")
            st.markdown(f"**Risk band:** :{color}[{band}]")
        with m2:
            st.markdown(f"**Recommendation:** {rec}")

        if st.button("Explain this prediction in plain English"):
            with st.spinner("Computing SHAP drivers and generating explanation..."):
                drivers = explain_single(explainer, pred["row_scaled"], feature_names, top_n=5)
                patient_context = {
                    "age": int(pred["row"]["age"].iloc[0]),
                    "lead_days": int(pred["row"]["lead_days"].iloc[0]),
                    "sms_received": bool(pred["row"]["sms_received"].iloc[0]),
                    "comorbidity_score": int(pred["row"]["comorbidity_score"].iloc[0]),
                }
                explanation = llm_client.explain_prediction(hf_client, risk_pct, drivers, patient_context)
                st.info(explanation)

            with st.expander("Raw SHAP values (fallback / detail view)"):
                for d in drivers:
                    st.write(f"- **{d['label']}**: {d['direction']} (SHAP = {d['shap_value']:+.4f})")

# TAB 3 - Model Performance
with tab3:
    st.subheader("Evaluation metrics")
    m1, m2, m3 = st.columns(3)
    m1.metric("Accuracy", f"{metrics['accuracy']*100:.1f}%")
    m2.metric("ROC-AUC (test)", f"{metrics['roc_auc']:.3f}")
    m3.metric("ROC-AUC (CV, train)", f"{metrics['cv_roc_auc']:.3f}")

    st.divider()
    st.subheader("Evaluation plots")
    p1, p2, p3 = st.columns(3)
    cm_path = os.path.join(PLOTS_DIR, "confusion_matrix.png")
    roc_path = os.path.join(PLOTS_DIR, "roc_curve.png")
    fi_path = os.path.join(PLOTS_DIR, "feature_importance.png")
    if os.path.exists(cm_path):
        p1.image(cm_path, caption="Confusion Matrix", width='stretch')
    if os.path.exists(roc_path):
        p2.image(roc_path, caption="ROC Curve", width='stretch')
    if os.path.exists(fi_path):
        p3.image(fi_path, caption="Feature Importance", width='stretch')

    st.divider()
    st.subheader("Best hyperparameters")
    st.json(metrics["best_params"])

    st.divider()
    st.subheader("Methodology")
    st.table(pd.DataFrame([
        {"Step": "Preprocessing", "Detail": "Clean raw CSV, engineer 12 features, drop invalid rows"},
        {"Step": "Split", "Detail": "80/20 stratified train/test split"},
        {"Step": "Scaling", "Detail": "StandardScaler fit on training set only"},
        {"Step": "Class balance", "Detail": "Class weighting via scale_pos_weight (no resampling)"},
        {"Step": "Model", "Detail": "XGBoost classifier"},
        {"Step": "Tuning", "Detail": "GridSearchCV with StratifiedKFold, optimising ROC-AUC"},
        {"Step": "Evaluation", "Detail": "Untouched, non-resampled test set"},
        {"Step": "Explainability", "Detail": "SHAP TreeExplainer, per-prediction and batch-level"},
    ]))

# TAB 4 - AI Insights
with tab4:
    st.subheader("Ask a question about the data")
    st.caption("Answers are grounded ONLY in the precomputed data summary below - the model never estimates numbers it wasn't given.")

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    example_cols = st.columns(3)
    example_qs = [
        "What is the overall no-show rate?",
        "Which day of week has the most no-shows?",
        "Does receiving an SMS reminder change the no-show rate?",
    ]
    clicked_q = None
    for col, q in zip(example_cols, example_qs):
        if col.button(q):
            clicked_q = q

    for role, msg in st.session_state.chat_history:
        with st.chat_message(role):
            st.write(msg)

    user_q = st.chat_input("Ask about the dataset...") or clicked_q
    if user_q:
        st.session_state.chat_history.append(("user", user_q))
        with st.chat_message("user"):
            st.write(user_q)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = llm_client.answer_question(hf_client, user_q, summary)
                st.write(answer)
        st.session_state.chat_history.append(("assistant", answer))

    st.divider()
    st.subheader("Executive summary")
    if st.button("Generate Summary"):
        with st.spinner("Generating weekly executive summary..."):
            exec_summary = llm_client.weekly_summary(hf_client, summary)
            st.success(exec_summary)

    with st.expander("Raw data summary (JSON) - fallback / detail view"):
        st.json(summary)
