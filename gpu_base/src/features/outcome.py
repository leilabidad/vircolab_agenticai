import pandas as pd

def add_outcome_from_admissions(df, admissions_path):
    """
    Add 'outcome' column to df based on 'hospital_expire_flag' from admissions CSV.

    Parameters:
        df (pd.DataFrame): main dataset containing 'subject_id'
        admissions_path (str or Path): path to admissions.csv.gz file

    Returns:
        pd.DataFrame: df with 'outcome' column added
    """
    # Load admissions with hospital_expire_flag
    admissions = pd.read_csv(admissions_path, usecols=["subject_id", "hospital_expire_flag"], compression="gzip")

    # Merge with main df on subject_id
    df = df.merge(admissions, on="subject_id", how="left")

    # Set outcome based on hospital_expire_flag
    df["outcome"] = df["hospital_expire_flag"].fillna(0).astype(int)

    # Remove hospital_expire_flag column
    df.drop(columns=["hospital_expire_flag"], inplace=True)

    return df
