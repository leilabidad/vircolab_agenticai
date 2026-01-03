from .features.demographic import build_demographic_features
from .features.ed import build_ed_features
from .features.chexpert import build_chexpert_features
from .features.split import build_split_features
from .features.metadata import build_metadata_features
from .features.outcome import add_outcome_from_admissions
from .note_generator import load_note_gpu
import pandas as pd
from pathlib import Path
import time
from tqdm import tqdm

def build_dataset(cfg, chunk_size=200):
    start_time = time.time()
    paths = cfg["paths"]

    demographic = build_demographic_features(cfg)
    ed = build_ed_features(cfg)

    if not demographic.empty and not ed.empty:
        mimic = demographic.merge(ed, on="subject_id", how="left")
    elif not demographic.empty:
        mimic = demographic.copy()
    elif not ed.empty:
        mimic = ed.copy()
    else:
        mimic = pd.DataFrame()

    chexpert = build_chexpert_features(cfg)
    split = build_split_features(cfg)
    metadata = build_metadata_features(cfg)

    valid_studies = metadata["study_id"].unique() if not metadata.empty else []

    if not chexpert.empty:
        chexpert = chexpert[chexpert["study_id"].isin(valid_studies)]
    if not split.empty:
        split = split[split["study_id"].isin(valid_studies)]
    if not mimic.empty:
        mimic = mimic[mimic["study_id"].isin(valid_studies)]

    df_all = chexpert
    if not split.empty:
        df_all = df_all.merge(split, on="study_id", how="left")
    if not metadata.empty:
        df_all = df_all.merge(metadata, on="study_id", how="left")
    if not mimic.empty:
        df_all = df_all.merge(mimic, on="subject_id", how="left")

    # cleanup ستون‌ها
    if "study_id_x" in df_all.columns:
        df_all.rename(columns={"study_id_x": "study_id"}, inplace=True)
    if "study_id_y" in df_all.columns:
        df_all.drop(columns=["study_id_y"], inplace=True)

    columns_to_write = df_all.columns.tolist()

    print(f"[INFO] Columns in dataset: {columns_to_write}")
    proceed = input("Do you want to continue with these columns? (y/n): ")
    if proceed.lower() != "y":
        raise SystemExit("Aborted by user")

    if not df_all.empty:
        df_all["path_clinical_note"] = load_note_gpu(df_all[["subject_id", "study_id"]], paths["notes"])

        df_all["outcome"] = add_outcome_from_admissions(df_all, cfg["paths"]["admissions_features"])["outcome"]


        img_cols = [c for c in ["path_img_fr", "path_img_la", "path_clinical_note"] if c in df_all.columns]
        if img_cols:
            df_all = df_all[df_all[img_cols].notna().all(axis=1)]

        df_all = df_all.drop_duplicates("study_id")

        # اطمینان از ترتیب ستون‌ها
        final_cols = ["study_id", "subject_id"] + [c for c in df_all.columns if c not in ["study_id", "subject_id", "outcome"]] + ["outcome"]
        df_all = df_all[final_cols]

    output_path = paths["output"]
    first_chunk = True
    for start in tqdm(range(0, len(df_all), chunk_size), desc="Writing dataset chunks"):
        chunk_df = df_all.iloc[start:start+chunk_size]
        chunk_df.to_csv(output_path, mode="w" if first_chunk else "a",
                        header=first_chunk, index=False, columns=df_all.columns)
        first_chunk = False

    end_time = time.time()
    print(f"[INFO] Dataset prepared in {end_time - start_time:.2f} seconds")
    return df_all
