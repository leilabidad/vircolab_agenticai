import cudf
import pandas as pd
from .note_generator import build_note_gpu, load_note_gpu
from .utils import progress

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","Pleural Effusion","Pleural Other",
    "Pneumonia","Pneumothorax","Support Devices"
]

def build_dataset(chexpert, split, frontal, lateral, cfg):
    patients = pd.read_csv(cfg["paths"]["patients_features"], usecols=['subject_id','gender','anchor_age'])
    admissions = pd.read_csv(cfg["paths"]["admissions_features"], usecols=['subject_id','admittime','hospital_expire_flag'])
    cxr = pd.read_csv(cfg["paths"]["metadata"], usecols=['subject_id','study_id','StudyDate','StudyTime'])

    patients.rename(columns={'anchor_age':'age'}, inplace=True)
    admissions['admittime'] = pd.to_datetime(admissions['admittime'])
    cxr['study_datetime'] = pd.to_datetime(
        cxr['StudyDate'].astype(str) + cxr['StudyTime'].astype(str).str.zfill(6),
        format="%Y%m%d%H%M%S",
        errors="coerce"
    )

    base = admissions.merge(patients, on='subject_id', how='left')
    base = base[(base['age'] >= 0) & (base['age'] <= 120)]
    base['outcome'] = base['hospital_expire_flag']

    mimic_df = cxr.merge(base[['subject_id','age','gender','outcome']], on='subject_id', how='left')
    mimic_df = mimic_df[['study_id','subject_id','age','gender','outcome']]
    mimic_df = mimic_df.dropna(subset=['study_id','subject_id'])
    mimic_df = mimic_df.drop_duplicates(subset='study_id')
    mimic_df['gender'] = mimic_df['gender'].map({'M':1,'F':0})

    chexpert_gpu = cudf.from_pandas(chexpert)
    split_gpu = cudf.from_pandas(split)
    mimic_gpu = cudf.from_pandas(mimic_df)

    df = chexpert_gpu[["subject_id","study_id"] + LABELS]
    df = df.merge(split_gpu[["study_id","split"]], on="study_id", how="left")
    df = df.merge(frontal, on="study_id", how="left")
    df = df.merge(lateral, on="study_id", how="left")
    df = df.merge(mimic_gpu, on="study_id", how="left")

    df_cpu = df.to_pandas()

    # رفع suffix و duplicate column
    for col in list(df_cpu.columns):
        if col.endswith("_x") or col.endswith("_y"):
            new_col = col[:-2]
            if new_col not in df_cpu.columns:
                df_cpu.rename(columns={col: new_col}, inplace=True)
            else:
                df_cpu.drop(columns=[col], inplace=True)
    df_cpu = df_cpu.loc[:,~df_cpu.columns.duplicated()]

    if cfg["clinical_note"]["mode"] == "fake":
        notes = build_note_gpu(df_cpu, LABELS, cfg)
    else:
        notes = load_note_gpu(df_cpu, cfg["paths"]["notes"])
    df_cpu["path_clinical_note"] = notes

    df_cpu["outcome"] = df_cpu["outcome"].fillna(0).astype(int)

    for col in ["path_img_fr","path_img_la"]:
        if col not in df_cpu.columns:
            df_cpu[col] = None

    if not cfg["filtering"]["keep_if_missing_all"]:
        cols = ["path_clinical_note", "path_img_fr", "path_img_la"]
        mask = (
            df_cpu[cols].notna().all(axis=1) &
            df_cpu[cols].apply(lambda c: c.astype(str).str.strip() != "").all(axis=1)
        )
        df_cpu = df_cpu[mask]

    return cudf.from_pandas(df_cpu.drop_duplicates("study_id"))
