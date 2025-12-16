import pandas as pd
from .note_generator import build_note, load_note
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
        if cfg["clinical_note"]["mode"] == "fake":
            notes.append(build_note(row, LABELS, cfg))
        else:
            notes.append(load_note(row, cfg["paths"]["notes"]))





    df["path_clinical_note"] = notes
    df["outcome"] = df[LABELS].eq(1).any(axis=1).astype(int)


    if not cfg["filtering"]["keep_if_missing_all"]:
        cols = ["path_clinical_note", "path_img_fr", "path_img_la"]

        mask = (
            df[cols].notna().all(axis=1) &
            df[cols].apply(lambda c: c.astype(str).str.strip() != "").all(axis=1)
        )

        df = df[mask]



    return df.drop_duplicates("study_id")
