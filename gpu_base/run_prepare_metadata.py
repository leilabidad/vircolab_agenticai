import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os
from src.vision.utils import load_cfg


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
    df.columns = df.columns.str.strip()
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
               [["study_id", col, "subject_id"]]  # keep subject_id
        )

        view_dfs[col] = tmp

    assert len(view_dfs) == 2, "Frontal or lateral images are completely missing"

    # Merge frontal + lateral, keep subject_id from frontal
    meta = (
        view_dfs["path_img_fr"]
        .merge(view_dfs["path_img_la"][["study_id", "path_img_la"]], on="study_id", how="inner")
        .sort_values("study_id")
        .reset_index(drop=True)
    )

    meta.columns = meta.columns.str.strip()
    assert not meta.empty, "No studies with both frontal and lateral images found"

    output_csv = Path(output_csv)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    meta.to_csv(output_csv, index=False)

    print("===================================")
    print(f"Total output rows: {len(meta)}")
    print(f"CSV saved to: {output_csv}")
    print("===================================")

    return meta


if __name__ == "__main__":

    cfg = load_cfg("./config/config.yaml")

    output_csv = Path(cfg["paths"]["mimic_prepared_csv"])

    build_metadata_features(cfg, str(output_csv))
