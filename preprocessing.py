import os
import numpy as np
import pandas as pd

RAW_PATH= os.path.join("data", "raw_appointments.csv")
PROCESSED_PATH=os.path.join("data", "processed_appointments.csv")

def load_data(path:str)-> pd.DataFrame:
    df= pd.read_csv(path)
    return df

def standardise_columns(df:pd.DataFrame)->pd.DataFrame:
    df= df.copy()
    new_cols=(
        df.columns
        .str.strip()
        .str.lower()
        .str.replace("-","_", regex=False)
        .str.replace(" ", "_", regex=False)
    )
    df.columns = new_cols

    rename_map={
        "hipertension": "hypertension",
        "handcap": "handicap",
        "patientid": "patient_id",
        "appointmentid": "appointment_id",
        "scheduledday": "scheduled_day",
        "appointmentday": "appointment_day",
        "sms_received": "sms_received",
    }
    df = df.rename(columns=rename_map)
    return df

def parse_dates(df: pd.DataFrame)-> pd.DataFrame:
    df = df.copy()
    df["scheduled_day"] = pd.to_datetime(df["scheduled_day"], utc =True).dt.tz_localize(None)
    df["appointment_day"] = pd.to_datetime(df["appointment_day"], utc =True).dt.tz_localize(None)
    return df

def encode_target(df: pd.DataFrame)-> pd.DataFrame:
    df=df.copy()
    df["no_show"] = df["no_show"].map({"Yes":1, "No":0}).astype(int)
    return df

def drop_invalid_rows(df:pd.DataFrame)-> pd.DataFrame:
    df=df.copy()

    before = len(df)
    lead_days_raw = (df["appointment_day"].dt.normalize() - df["scheduled_day"].dt.normalize()).dt.days
    df=df[lead_days_raw>=0]
    df=df[(df["age"]>=0) & (df["age"]<=115)]
    after = len(df)
    print(f"[drop_invalid_rows] remove {before - after} rows ({before}->{after})")
    return df

def engineer_features(df:pd.DataFrame)-> pd.DataFrame:
    df = df.copy()
    df["lead_days"]=(
        df["appointment_day"].dt.normalize() - df["scheduled_day"].dt.normalize()
    ).dt.days

    df["appt_day_of_week"] = df["appointment_day"].dt.dayofweek
    df["appt_day_of_month"] = df["appointment_day"].dt.day
    df["scheduled_hour"] = df["scheduled_day"].dt.hour
    df["is_weekend"] = df["appt_day_of_week"].isin([5,6]).astype(int)

    def bucket_age(age):
        if age <=12:
            return "child"
        elif age<=19:
            return "teen"
        elif age<= 35:
            return "young_adult"
        elif age<=60:
            return "adult"
        else:
            return "senior"
        
    df["age_group"] = df["age"].apply(bucket_age)

    # Sum of binsary health flags
    df["comorbidity_score"] = (
        df["hypertension"] + df["diabetes"] + df["alcoholism"] + df["handicap"].clip(upper=1)
    )

    # Total appointments of each patient (history)
    df["patient_appt_count"] = df.groupby("patient_id")["appointment_id"].transform("count")

    # NOTE: neighbourhood_noshowrate is intentionally NOT computed here.
    # Computing it over the full dataset (before train/test split) would leak
    # test-set outcomes into a feature the model trains on. It's computed in
    # train.py instead, using the training split only, and mapped onto test.
    return df

def encode_categorials(df: pd.DataFrame)-> pd.DataFrame:
    df= df.copy()
    df["gender_encoded"] = df["gender"].map({"F": 1, "M" : 0})
    age_group_order = {"child":0, "teen":1, "young_adult":2, "adult":3,"senior":4}
    df["age_group_encoded"] = df["age_group"].map(age_group_order)
    return df

def run_pipeline(raw_path:str = RAW_PATH, out_path:str = PROCESSED_PATH)->pd.DataFrame:
    df= load_data(raw_path)
    print(f"[load_raw] loaded {len(df)} rows, {df.shape[1]} columns")

    df = standardise_columns(df)
    df = parse_dates(df)
    df = encode_target(df)
    df = drop_invalid_rows(df)
    df = engineer_features(df)
    df= encode_categorials(df)

    os.makedirs(os.path.dirname(out_path), exist_ok= True)
    df.to_csv(out_path, index=False)
    return df

if __name__ == "__main__":
    df= run_pipeline()
    print("\nPREPROCESS.PY SUMMARY")
    print(f"Final row count: {len(df)}")
    print(f"Feature matrix shape: {df.shape}")
    print(f"Target class balance (no_show): ")
    print(df["no_show"].value_counts(normalize=True).rename("proportion"))
    print(f"\nSaved processed file to: {PROCESSED_PATH}")