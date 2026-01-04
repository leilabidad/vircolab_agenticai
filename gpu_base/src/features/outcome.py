import pandas as pd

def add_outcome_from_admissions(df, admissions_path):
    admissions = pd.read_csv(
        admissions_path,
        usecols=["subject_id", "hospital_expire_flag"],
        compression="gzip"
    )

    # Aggregate to one row per subject
    admissions = (
        admissions
        .groupby("subject_id", as_index=False)
        .hospital_expire_flag
        .max()
    )

    # --- Fix type mismatch ---
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    admissions["subject_id"] = admissions["subject_id"].astype(str).str.strip()
    # ------------------------

    # Merge outcome safely
    df = df.merge(admissions, on="subject_id", how="left")

    # Keep NaN where there is no hospital_expire_flag
    df["outcome"] = df["hospital_expire_flag"]
    df.drop(columns=["hospital_expire_flag"], inplace=True)

    return df
