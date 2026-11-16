# Medical Appointment No-Show Prediction

Predicts the probability that a patient will miss a scheduled appointment, explains each prediction with SHAP, and uses an LLM to narrate dataset-level insights in plain English. Includes an interactive Streamlit dashboard.

> **Demo project.** Trained on a public historical dataset and not validated on a real clinic's population. Predictions and AI-generated narration should not inform real clinical or scheduling decisions without further validation.

## Results

Evaluated on a stratified 20% hold-out set (22,105 appointments).

| Metric | Value |
|---|---|
| ROC-AUC (test) | **0.745** |
| ROC-AUC (3-fold CV, best model) | 0.751 |
| Accuracy | 80.2% (majority-class baseline: 79.8%) |
| No-show precision | 0.58 |
| No-show recall | 0.06 |

**Reading these numbers honestly:** the model ranks appointments by risk reasonably well (AUC 0.745, with test and CV scores close together, so no sign of overfitting). At the default 0.5 decision threshold, however, it flags very few no-shows (recall 0.06), because only 20.2% of appointments are no-shows and no imbalance correction was applied. Accuracy is therefore not a meaningful headline metric here. See [Limitations and next steps](#limitations-and-next-steps).

### What drives no-shows

Top features by XGBoost importance:

| Feature | Importance |
|---|---|
| `lead_days` (days between booking and appointment) | 0.341 |
| `sms_received` | 0.082 |
| `age` | 0.076 |
| `patient_appt_count` (patient history) | 0.060 |
| `alcoholism` | 0.053 |

Lead time is by far the strongest signal, about four times the importance of the next feature.

## Dataset

[Medical Appointment No-Shows](https://www.kaggle.com/datasets/joniarroba/noshowappointments) (Kaggle): 110,527 appointments, 110,521 after cleaning. 20.2% are no-shows. The raw CSV is not included in this repo; download it and place it in `data/`.

## Approach

1. **Preprocessing** (`preprocessing.py`): cleans the raw data and engineers features (23 columns in total), such as lead time, day of week, comorbidity indicators and patient appointment history.
2. **Split and scaling:** 80/20 stratified train/test split. `StandardScaler` is fit on the training set only.
3. **Tuning** (`train.py`): XGBoost classifier tuned with `GridSearchCV` over 32 parameter combinations (3-fold stratified CV, 96 fits), optimizing ROC-AUC.
   Best parameters: `n_estimators=200`, `max_depth=5`, `learning_rate=0.1`, `subsample=0.8`, `colsample_bytree=1.0`.
4. **Explainability** (`shap_explainer.py`): SHAP `TreeExplainer` for per-patient and global explanations.
5. **AI insights** (`llm_client.py`): Llama 3.1 8B Instruct (Mistral 7B as fallback) via the Hugging Face Inference API. Prompts are restricted to numbers from a precomputed summary, and the app falls back to raw SHAP values and data summaries when no token is set.

## Dashboard

Run `streamlit run app.py`. Four tabs:

1. **Dataset Insights:** no-show rates by day of week, lead time, age group and SMS reminder status.
2. **Risk Predictor:** risk percentage, risk band and recommended action for a single appointment.
3. **Model Performance:** evaluation plots and metrics.
4. **AI Insights:** LLM-written summary of the patterns in the data.

## Getting started

```bash
pip install -r requirements.txt
```

1. Download the Kaggle dataset into `data/`.
2. Optional: copy `.env.example` to `.env` and add a Hugging Face token to enable AI narration.
3. Train the model:
   ```bash
   python train.py
   ```
   This saves the model and scaler to `model/` and evaluation plots to `plots/`.
4. Launch the dashboard:
   ```bash
   streamlit run app.py
   ```

## Project structure

```
app.py              Streamlit dashboard
train.py            Training, tuning and evaluation
preprocessing.py    Cleaning and feature engineering
shap_explainer.py   SHAP explanations
llm_client.py       LLM narration (Hugging Face Inference API)
data/               Raw dataset (not included)
model/              Saved model artifacts
plots/              Evaluation and feature-importance plots
```

## Limitations and next steps

- **Low no-show recall at the default threshold.** Planned fixes: tune the decision threshold on a precision-recall curve, add `scale_pos_weight`, and report precision, recall and F1 at the chosen operating point. SMOTE was tried but skipped because of a threadpool issue in the training environment.
- **Public data only.** Not validated on a real clinic's population, and patient behaviour varies by region and clinic.
- **Moderate predictive signal.** An AUC of 0.745 reflects limited information in the available features. Richer data (appointment type, prior no-show history by patient, weather, transport access) would likely help.
- **LLM narration** on the free Hugging Face tier can be slow after idle periods.

## Tech stack

Python, pandas, NumPy, scikit-learn, XGBoost, SHAP, Streamlit, Hugging Face Inference API.
