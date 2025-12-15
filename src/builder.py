import pandas as pd
from .note_generator import build_note
from .utils import progress

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","Pleural Effusion","Pleural Other",
    "Pneumonia","Pneumothorax","Support Devices"
]

def build_dataset(chexpert, split, frontal, lateral, cfg):
    df = chexpert[["subject_id","study_id"] + LABELS].copy()
    df = df.merge(split[["study_id","split"]], on="study_id", how="left")
    df = df.merge(frontal, on="study_id", how="left")
    df = df.merge(lateral, on="study_id", how="left")

    notes = []
    for _, row in progress(df.iterrows(), cfg["runtime"]["progress_bar"]):
        notes.append(build_note(row, LABELS, cfg))

    df["path_clinical_note"] = notes
    df["outcome"] = df[LABELS].eq(1).any(axis=1).astype(int)

    if not cfg["filtering"]["keep_if_missing_all"]:
        df = df[
            df["path_clinical_note"].notna() |
            df["path_img_fr"].notna() |
            df["path_img_la"].notna()
        ]

    return df.drop_duplicates("study_id")
