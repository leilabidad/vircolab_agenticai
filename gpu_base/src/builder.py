import cudf
import pandas as pd
from .note_generator import build_note_gpu, load_note_gpu
from .features.demographic import build_demographic_features
from .features.ed import build_ed_features

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","Pleural Effusion","Pleural Other",
    "Pneumonia","Pneumothorax","Support Devices"
]

def _cleanup_columns(df):
    for col in list(df.columns):
        if col.endswith("_x") or col.endswith("_y"):
            base = col[:-2]
            if base not in df.columns:
                df.rename(columns={col: base}, inplace=True)
            else:
                df.drop(columns=[col], inplace=True)
    return df.loc[:, ~df.columns.duplicated()]

def merge_in_chunks(df, other, on, how="left", chunk_size=5000):
    out = []
    for i in range(0, len(df), chunk_size):
        part = df.iloc[i:i+chunk_size]
        out.append(part.merge(other, on=on, how=how))
        print(f"############################## Merged chunk ordinary {i} to {i+chunk_size} / {len(df)}")
    return pd.concat(out, ignore_index=True)

def build_notes_in_chunks(df, cfg, chunk_size=2000):
    notes = []
    for i in range(0, len(df), chunk_size):
        part = df.iloc[i:i+chunk_size]
        # if cfg["clinical_note"]["mode"] == "fake":
        #     notes.extend(build_note_gpu(part, LABELS, cfg))
        # else:
        #     notes.extend(load_note_gpu(part, cfg["paths"]["notes"]))
        notes.extend(load_note_gpu(part, cfg["paths"]["notes"]))
        print(f"############################## Merged chunk NOTES {i} to {i+chunk_size} / {len(df)}")
    return notes

def build_dataset(chexpert, split, frontal, lateral, cfg):
    demographic = build_demographic_features(cfg)
    ed = build_ed_features(cfg)
    mimic = demographic.merge(ed, on="subject_id", how="left")

    chexpert_gpu = cudf.from_pandas(chexpert)
    split_gpu = cudf.from_pandas(split)

    df = chexpert_gpu[["subject_id","study_id"] + LABELS]
    df = df.merge(split_gpu[["study_id","split"]], on="study_id", how="left")
    df = df.merge(frontal, on="study_id", how="left")
    df = df.merge(lateral, on="study_id", how="left")

    df = df.to_pandas()
    df = merge_in_chunks(df, mimic, on="subject_id", how="left")
    df = _cleanup_columns(df)

    df["path_clinical_note"] = build_notes_in_chunks(df, cfg)
    df["outcome"] = df["outcome"].fillna(0).astype(int)

    for col in ["path_img_fr","path_img_la"]:
        if col not in df.columns:
            df[col] = None

    if not cfg["filtering"]["keep_if_missing_all"]:
        cols = ["path_clinical_note","path_img_fr","path_img_la"]
        mask = (
            df[cols].notna().all(axis=1) &
            df[cols].astype(str).apply(lambda c: c.str.strip() != "").all(axis=1)
        )
        df = df[mask]

    return df.drop_duplicates("study_id")
