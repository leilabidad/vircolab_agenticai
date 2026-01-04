import pandas as pd
from pathlib import Path
from src.vision.utils import load_cfg
from .features.chexpert import build_chexpert_features
from .features.split import build_split_features
from .features.demographic import build_demographic_features
from .features.ed import build_ed_features
from .features.outcome import add_outcome_from_admissions
from .note_generator import load_note_gpu


def build_dataset(cfg):
    paths = cfg["paths"]

    metadata_csv = Path(paths["mimic_prepared_csv"])
    output_csv = Path(paths["mimic_prepared_csv"])

    print("[INFO] Loading prepared metadata CSV...")
    df = pd.read_csv(metadata_csv)
    assert not df.empty, "Prepared metadata CSV is empty"


    
    ##########################################
    print("[INFO] Adding CheXpert features...")
    chexpert = build_chexpert_features(cfg)
    chexpert = chexpert.groupby("subject_id", as_index=False).first()
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    chexpert["subject_id"] = chexpert["subject_id"].astype(str).str.strip()
    
    # Remove columns from chexpert that conflict with df except the merge key
    cols_to_drop = [c for c in ["study_id"] if c in chexpert.columns]
    chexpert = chexpert.drop(columns=cols_to_drop)
    df = df.merge(chexpert, on="subject_id", how="left")
    cols = ["study_id", "subject_id"] + [c for c in df.columns if c not in ["study_id", "subject_id"]]
    df = df[cols]

    
    ########################################3df = df.merge(chexpert, on="subject_id", how="left")

    
    ##########################################
    
    print("[INFO] Adding demographic features...")
    demographic = build_demographic_features(cfg)
    demographic = demographic.groupby("subject_id", as_index=False).first()
    df["subject_id"] = df["subject_id"].astype(str).str.strip()
    demographic["subject_id"] = demographic["subject_id"].astype(str).str.strip()
    cols_to_drop = [c for c in ["study_id"] if c in demographic.columns]
    demographic = demographic.drop(columns=cols_to_drop)

    df = df.merge(demographic, on="subject_id", how="left")
    cols = ["study_id", "subject_id"] + [c for c in df.columns if c not in ["study_id", "subject_id"]]
    df = df[cols]
    
    ##########################################
    print("[INFO] Adding clinicalnotes ...")
    df["path_clinical_note"] = load_note_gpu(df[["subject_id", "study_id"]], paths["notes"])
    

    ##########################################


    print("[INFO] Adding outcome column...")
    df = add_outcome_from_admissions(df, paths["admissions_features"])

    # Ensure outcome is the last column
    cols = [c for c in df.columns if c != "outcome"] + ["outcome"]

    # Filter rows missing images or notes
    img_cols = [c for c in ["path_img_fr", "path_img_la", "path_clinical_note", "outcome"] if c in df.columns]
    if img_cols:
        df = df[df[img_cols].notna().all(axis=1)]

    # Ensure column order: study_id, subject_id, other columns, outcome last
    #final_cols = ["study_id", "subject_id"] + [c for c in df.columns if c not in ["study_id", "subject_id", "outcome"]] + ["outcome"]
    #df = df[final_cols]



    df = df[cols]

    #df = df.drop_duplicates("study_id")

    print(f"[INFO] Final dataset rows: {len(df)}")

    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)

    print(f"[INFO] Final dataset saved to {output_csv}")
    return df


# if __name__ == "__main__":
#     cfg = load_cfg("./config/config.yaml")
#     build_dataset(cfg)
