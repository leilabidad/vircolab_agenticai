from .features.demographic import build_demographic_features
from .features.ed import build_ed_features
from .features.chexpert import build_chexpert_features
from .features.split import build_split_features
from .features.metadata import build_metadata_features
from .note_generator import load_note_gpu
import pandas as pd
from pathlib import Path
import time

def build_dataset(cfg, chunk_size=200):
    start_time = time.time()

    demographic = build_demographic_features(cfg)
    ed = build_ed_features(cfg)
    mimic = demographic.merge(ed, on="subject_id", how="left")

    chexpert = build_chexpert_features(cfg)
    split = build_split_features(cfg)
    metadata = build_metadata_features(cfg)

    # فقط study_id هایی که عکس دارن
    valid_studies = metadata["study_id"].unique()

    chexpert = chexpert[chexpert["study_id"].isin(valid_studies)]
    split = split[split["study_id"].isin(valid_studies)]
    mimic = mimic[mimic["study_id"].isin(valid_studies)]

    df_all = (
        chexpert
        .merge(split, on="study_id", how="left")
        .merge(metadata, on="study_id", how="left")
        .merge(mimic, on="subject_id", how="left")
    )

    if "study_id_x" in df_all.columns:
        df_all.rename(columns={"study_id_x": "study_id"}, inplace=True)
    if "study_id_y" in df_all.columns:
        df_all.drop(columns=["study_id_y"], inplace=True)

    columns_to_write = ["study_id", "subject_id", "path_img_fr", "path_img_la",
                        "path_clinical_note", "outcome"]

    print(f"[INFO] Columns in dataset: {df_all.columns.tolist()}")
    proceed = input("Do you want to continue with these columns? (y/n): ")
    if proceed.lower() != "y":
        raise SystemExit("Aborted by user")

    df_all["path_clinical_note"] = load_note_gpu(df_all[["subject_id", "study_id"]], cfg["paths"]["notes"])
    df_all["outcome"] = df_all["outcome"].fillna(0).astype(int)

    df_all = df_all[df_all[["path_img_fr", "path_img_la", "path_clinical_note"]].notna().all(axis=1)]
    df_all = df_all.drop_duplicates("study_id")
    df_all = df_all[columns_to_write]

    output_path = cfg["paths"]["output"]
    first_chunk = True
    for start in range(0, len(df_all), chunk_size):
        chunk_df = df_all.iloc[start:start+chunk_size]
        chunk_df.to_csv(output_path, mode="w" if first_chunk else "a",
                        header=first_chunk, index=False, columns=columns_to_write)
        first_chunk = False
        print(f"[INFO] Saved chunk {start} to {start+len(chunk_df)} / {len(df_all)}")

    end_time = time.time()
    print(f"[INFO] Dataset prepared in {end_time - start_time:.2f} seconds")
    return df_all
