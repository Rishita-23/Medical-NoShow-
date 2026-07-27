import argparse
import os
import pickle
import time
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
from preprocessing import PROCESSED_PATH, run_pipeline

MODEL_DIR = "model"
PLOTS_DIR = "plots"
MODEL_PATH = os.path.join(MODEL_DIR, "noshow_model.pkl")

FEATURE_COLUMNS = [
    "age",
    "scholarship",
    "hypertension",
    "diabetes",
    "alcoholism",
    "handicap",
    "sms_received",
    "lead_days",
    "appt_day_of_week",
    "appt_day_of_month",
    "scheduled_hour",
    "is_weekend",
    "comorbidity_score",
    "patient_appt_count",
    "neighbourhood_noshowrate",
    "gender_encoded",
    "age_group_encoded",
]
TARGET_COLUMNS = "no_show"


def load_processed() -> pd.DataFrame:
    if not os.path.exists(PROCESSED_PATH):
        print(f"[train] {PROCESSED_PATH} not found, running preprocess pipeline first")
        return run_pipeline()
    return pd.read_csv(PROCESSED_PATH)


def build_params_grid(quick: bool) -> dict:
    if quick:
        return {
            "n_estimators": [100],
            "max_depth": [4],
            "learning_rate": [0.1],
            "subsample": [0.8],
        }
    return {
        "n_estimators": [150, 300],
        "max_depth": [3, 5, 7],
        "learning_rate": [0.01, 0.05, 0.1],
        "subsample": [0.8, 1.0],
    }


def main(quick: bool = False):
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(PLOTS_DIR, exist_ok=True)

    df = load_processed()

    df_train, df_test = train_test_split(
        df, test_size=0.2, stratify=df[TARGET_COLUMNS], random_state=42
    )
    df_train = df_train.copy()
    df_test = df_test.copy()

    # Compute neighbourhood no-show rate from the TRAINING split only, then
    # map it onto both splits. Doing this before the split (as the old code
    # did) would leak test-set outcomes into a feature the model trains on.
    neigh_rate_map = df_train.groupby("neighbourhood")[TARGET_COLUMNS].mean()
    global_rate = df_train[TARGET_COLUMNS].mean()
    df_train["neighbourhood_noshowrate"] = df_train["neighbourhood"].map(neigh_rate_map)
    df_test["neighbourhood_noshowrate"] = (
        df_test["neighbourhood"].map(neigh_rate_map).fillna(global_rate)
    )

    X_train = df_train[FEATURE_COLUMNS]
    X_test = df_test[FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMNS]
    y_test = df_test[TARGET_COLUMNS]
    print(f"[train] train shape: {X_train.shape}, test shape: {X_test.shape}")

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    print(f"[train] training class balance (unmodified, no resampling): {np.bincount(y_train)}")

    param_grid = build_params_grid(quick)
    cv = StratifiedKFold(n_splits=3 if quick else 5, shuffle=True, random_state=42)

    neg, pos = np.bincount(y_train)
    scale_pos_weight = neg / pos
    print(f"[train] scale_pos_weight = {scale_pos_weight:.3f}")

    base_model = XGBClassifier(
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
    )
    grid = GridSearchCV(
        base_model,
        param_grid=param_grid,
        scoring="roc_auc",
        cv=cv,
        n_jobs=-1,
        verbose=1,
    )
    print(f"[train] starting GridSearchCV ({'quick' if quick else 'full'} grid)")
    t0 = time.time()
    grid.fit(X_train_scaled, y_train)
    print(f"[train] grid search done in {time.time() - t0:.1f}s")
    print(f"[train] best params: {grid.best_params_}")
    print(f"[train] best CV ROC-AUC: {grid.best_score_:.4f}")

    best_model = grid.best_estimator_

    # on untouched data
    y_pred = best_model.predict(X_test_scaled)
    y_proba = best_model.predict_proba(X_test_scaled)[:, 1]

    accuracy = accuracy_score(y_test, y_pred)
    roc_auc = roc_auc_score(y_test, y_proba)
    cm = confusion_matrix(y_test, y_pred)

    print("\nTest set evaluation")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("Confusion matrix:")
    print(cm)

# PLOTS
    # confusion matrix heatmap
    fig, ax = plt.subplots(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No-show=0", "No-show=1"], yticklabels=["No-show=0", "No-show=1"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "confusion_matrix.png"), dpi=150)
    plt.close(fig)

    # ROC curve
    fig, ax = plt.subplots(figsize=(5, 4))
    RocCurveDisplay.from_predictions(y_test, y_proba, ax=ax)
    ax.set_title("ROC Curve")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "roc_curve.png"), dpi=150)
    plt.close(fig)

    # Feature importance
    importances = best_model.feature_importances_
    imp_df = pd.DataFrame({"feature": FEATURE_COLUMNS, "importance": importances})
    imp_df = imp_df.sort_values("importance", ascending=False)

    fig, ax = plt.subplots(figsize=(7, 6))
    sns.barplot(data=imp_df, x="importance", y="feature", ax=ax, color="steelblue")
    ax.set_title("XGBoost Feature Importance")
    fig.tight_layout()
    fig.savefig(os.path.join(PLOTS_DIR, "feature_importance.png"), dpi=150)
    plt.close(fig)

    # ---- Persist everything the app needs ----
    artifact = {
        "model": best_model,
        "scaler": scaler,
        "feature_names": FEATURE_COLUMNS,
        "neighbourhood_noshowrate_map": neigh_rate_map.to_dict(),
        "neighbourhood_noshowrate_global": float(global_rate),
        "metrics": {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "confusion_matrix": cm.tolist(),
            "best_params": grid.best_params_,
            "cv_roc_auc": grid.best_score_,
        },
    }
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(artifact, f)

    top5 = imp_df.head(5)

    print("\ntrain.py summary")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"ROC-AUC:  {roc_auc:.4f}")
    print("Top 5 features by importance:")
    for _, row in top5.iterrows():
        print(f"  {row['feature']:<28} {row['importance']:.4f}")
    print(f"\nSaved model artifact to: {MODEL_PATH}")
    print(f"Saved plots to: {PLOTS_DIR}/")

    return artifact


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="Use a small grid for fast verification runs")
    args = parser.parse_args()
    main(quick=args.quick)