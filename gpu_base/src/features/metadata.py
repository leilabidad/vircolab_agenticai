import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import os

def _check_paths_exist(paths):
    results = []
    with ThreadPoolExecutor() as ex:
        for res in tqdm(ex.map(os.path.exists, paths), total=len(paths), desc="Checking images"):
            results.append(res)
    return results

def build_metadata_features(cfg):
    df = pd.read_csv(cfg["paths"]["metadata"], usecols=["subject_id", "study_id", "dicom_id", "ViewPosition"])
    assert len(df) > 0, "METADATA CSV IS EMPTY"

    base = Path(cfg["paths"]["images"])
    assert base.exists(), f"IMAGE BASE PATH DOES NOT EXIST: {base}"

    out = []
    for views, col in [(["PA", "AP"], "path_img_fr"), (["LATERAL"], "path_img_la")]:
        tmp = df[df["ViewPosition"].isin(views)].copy()
        if tmp.empty:
            continue

        tmp[col] = [str(base / ("p"+str(s)[:2]) / ("p"+str(s)) / ("s"+str(st)) / (str(d)+".jpg"))
                     for s, st, d in zip(tmp["subject_id"], tmp["study_id"], tmp["dicom_id"])]
        exists = _check_paths_exist(tmp[col])
        tmp = tmp[exists]
        if tmp.empty:
            continue

        tmp = tmp.sort_values(["study_id", "dicom_id"]).drop_duplicates("study_id")
        out.append(tmp[["study_id", col]])

    assert out, "NO IMAGES FOUND AT ALL — CHECK PATH LOGIC"

    meta = out[0]
    for t in out[1:]:
        meta = meta.merge(t, on="study_id", how="outer")

    return meta
