import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os


def _check_paths_exist(paths):
    with ThreadPoolExecutor() as ex:
        return list(
            tqdm(
                ex.map(os.path.exists, paths),
                total=len(paths),
                desc="Checking images"
            )
        )


def build_metadata_features(cfg, output_csv):
    df = pd.read_csv(
        cfg["paths"]["metadata"],
        usecols=["subject_id", "study_id", "dicom_id", "ViewPosition"]
    )
    assert not df.empty, "Metadata CSV is empty"

    base = Path(cfg["paths"]["images"])
    assert base.exists(), f"Image base path does not exist: {base}"

    view_map = {
        "path_img_fr": ["PA", "AP"],
        "path_img_la": ["LATERAL"]
    }

    view_dfs = {}

    for col, views in view_map.items():
        tmp = df[df["ViewPosition"].isin(views)].copy()
        if tmp.empty:
            continue

        tmp[col] = [
            str(base / f"p{str(s)[:2]}" / f"p{s}" / f"s{st}" / f"{d}.jpg")
            for s, st, d in zip(tmp.subject_id, tmp.study_id, tmp.dicom_id)
        ]

        exists = _check_paths_exist(tmp[col])
        tmp = tmp[exists]

        if tmp.empty:
            continue

        tmp = (
            tmp.sort_values(["study_id", "dicom_id"])
               .drop_duplicates("study_id")
               [["study_id", col]]
        )

        view_dfs[col] = tmp

    assert len(view_dfs) == 2, "Frontal or lateral images are completely missing"

    meta = (
        view_dfs["path_img_fr"]
        .merge(view_dfs["path_img_la"], on="study_id", how="inner")
        .sort_values("study_id")
        .reset_index(drop=True)
    )

    assert not meta.empty, "No studies with both frontal and lateral images found"
    meta = meta.drop_duplicates("study_id")

    meta.to_csv(output_csv, index=False)

    print("===================================")
    print(f"Total output rows: {len(meta)}")
    print(f"CSV saved to: {output_csv}")
    print("===================================")

    return meta

