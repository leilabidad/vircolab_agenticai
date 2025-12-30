import pandas as pd

def build_demographic_features(cfg):
    patients = pd.read_csv(cfg["paths"]["patients_features"], usecols=["subject_id","gender","anchor_age"])
    admissions = pd.read_csv(cfg["paths"]["admissions_features"], usecols=["subject_id","admittime","hospital_expire_flag"])
    cxr = pd.read_csv(cfg["paths"]["metadata"], usecols=["subject_id","study_id","StudyDate","StudyTime"])

    patients = patients.rename(columns={"anchor_age":"age"})
    admissions["admittime"] = pd.to_datetime(admissions["admittime"])

    cxr["study_datetime"] = pd.to_datetime(
        cxr["StudyDate"].astype(str) + cxr["StudyTime"].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    base = admissions.merge(patients, on="subject_id", how="left")
    base = base.query("0 <= age <= 120")
    base["outcome"] = base["hospital_expire_flag"]

    df = cxr.merge(base[["subject_id","age","gender","outcome"]], on="subject_id", how="left")
    df = df[["study_id","subject_id","study_datetime","age","gender","outcome"]]
    df = df.dropna(subset=["study_id","subject_id"])
    df = df.drop_duplicates("study_id")
    df["gender"] = df["gender"].map({"M":1,"F":0})

    return df
