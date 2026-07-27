from typing import List, Dict

import numpy as np
import pandas as pd
import shap

FEATURE_LABELS: Dict[str, str] = {
    "age": "Patient age",
    "scholarship": "Enrolled in welfare scholarship program",
    "hypertension": "Has hypertension",
    "diabetes": "Has diabetes",
    "alcoholism": "Has history of alcoholism",
    "handicap": "Has a handicap",
    "sms_received": "Received an SMS reminder",
    "lead_days": "Days between booking and appointment",
    "appt_day_of_week": "Day of the week of the appointment",
    "appt_day_of_month": "Day of the month of the appointment",
    "scheduled_hour": "Hour of day the appointment was booked",
    "is_weekend": "Appointment falls on a weekend",
    "comorbidity_score": "Number of chronic conditions (comorbidity score)",
    "patient_appt_count": "Total number of appointments this patient has booked",
    "neighbourhood_noshowrate": "Historical no-show rate for the patient's neighbourhood",
    "gender_encoded": "Patient gender",
    "age_group_encoded": "Patient age group",
}


def build_explainer(model) -> shap.TreeExplainer:
    return shap.TreeExplainer(model)


def _label(feature_name: str) -> str:
    return FEATURE_LABELS.get(feature_name, feature_name)


def explain_single(
    explainer: shap.TreeExplainer,
    X_row: np.ndarray,
    feature_names: List[str],
    top_n: int = 5,
) -> List[Dict]:
    X_row = np.asarray(X_row).reshape(1, -1)
    shap_values = explainer.shap_values(X_row)
    if isinstance(shap_values, list):  # some SHAP versions return a list for binary clf
        shap_values = shap_values[-1]
    values = np.asarray(shap_values).reshape(-1)

    order = np.argsort(-np.abs(values))[:top_n]

    drivers = []
    for idx in order:
        val = float(values[idx])
        drivers.append({
            "feature": feature_names[idx],
            "label": _label(feature_names[idx]),
            "shap_value": val,
            "direction": "increases risk" if val > 0 else "decreases risk",
            "magnitude": abs(val),
        })
    return drivers


def explain_batch(
    explainer: shap.TreeExplainer,
    X_batch: np.ndarray,
    feature_names: List[str],
    top_n: int = 10,
) -> List[Dict]:
    X_batch = np.asarray(X_batch)
    shap_values = explainer.shap_values(X_batch)
    if isinstance(shap_values, list):
        shap_values = shap_values[-1]
    values = np.asarray(shap_values)

    mean_abs = np.abs(values).mean(axis=0)
    mean_signed = values.mean(axis=0)

    order = np.argsort(-mean_abs)[:top_n]

    drivers = []
    for idx in order:
        drivers.append({
            "feature": feature_names[idx],
            "label": _label(feature_names[idx]),
            "mean_abs_shap": float(mean_abs[idx]),
            "mean_signed_shap": float(mean_signed[idx]),
            "direction": "increases risk" if mean_signed[idx] > 0 else "decreases risk",
        })
    return drivers


if __name__ == "__main__":
    import pickle

    with open("model/noshow_model.pkl", "rb") as f:
        artifact = pickle.load(f)

    model = artifact["model"]
    scaler = artifact["scaler"]
    feature_names = artifact["feature_names"]

    explainer = build_explainer(model)

    # Fabricated sample patient: long lead time, no SMS reminder, young adult, no comorbidities
    sample = pd.DataFrame([{
        "age": 28,
        "scholarship": 0,
        "hypertension": 0,
        "diabetes": 0,
        "alcoholism": 0,
        "handicap": 0,
        "sms_received": 0,
        "lead_days": 45,
        "appt_day_of_week": 4,
        "appt_day_of_month": 15,
        "scheduled_hour": 9,
        "is_weekend": 0,
        "comorbidity_score": 0,
        "patient_appt_count": 1,
        "neighbourhood_noshowrate": 0.20,
        "gender_encoded": 0,
        "age_group_encoded": 2,
    }])[feature_names]

    sample_scaled = scaler.transform(sample)
    drivers = explain_single(explainer, sample_scaled, feature_names, top_n=5)

    print("Sample patient: 28yo, 45-day lead time, no SMS reminder, no comorbidities")
    print("Top 5 SHAP drivers:")
    for d in drivers:
        print(f"  {d['label']:<55} {d['direction']:<16} (shap={d['shap_value']:+.4f})")

    # Now compare with a short-lead-time, SMS-reminded twin
    sample2 = sample.copy()
    sample2["lead_days"] = 1
    sample2["sms_received"] = 1
    sample2_scaled = scaler.transform(sample2)

    proba_long = model.predict_proba(scaler.transform(sample))[0, 1]
    proba_short = model.predict_proba(sample2_scaled)[0, 1]
    print(f"\nPredicted no-show risk (45-day lead, no SMS): {proba_long:.3f}")
    print(f"Predicted no-show risk (1-day lead, w/ SMS):   {proba_short:.3f}")
    print("Sanity check (longer lead time / no reminder should raise risk):",
          "PASS" if proba_long > proba_short else "FAIL")