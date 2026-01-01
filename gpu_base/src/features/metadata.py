import pandas as pd
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def _check_paths_exist(paths):
    with ThreadPoolExecutor() as ex:
        return list(ex.map(lambda p: Path(p).exists(), paths))

def build_metadata_features(cfg):
    df = pd.read_csv(
        cfg["paths"]["metadata"],
        usecols=["subject_id", "study_id", "dicom_id", "ViewPosition"]
    )

    assert len(df) > 0, "METADATA CSV IS EMPTY"

    base = Path(cfg["paths"]["images"])
    assert base.exists(), f"IMAGE BASE PATH DOES NOT EXIST: {base}"

    def build_paths(sub):
        return (
            base
            / ("p" + sub["subject_id"].astype(str).str[:2])
            / ("p" + sub["subject_id"].astype(str))
            / ("s" + sub["study_id"].astype(str))
            / (sub["dicom_id"].astype(str) + ".jpg")
        ).astype(str)

    out = []

    for views, col in [
        (["PA", "AP"], "path_img_fr"),
        (["LATERAL"], "path_img_la"),
    ]:
        tmp = df[df["ViewPosition"].isin(views)].copy()

        if tmp.empty:
            print(f"[WARN] NO {col} CANDIDATES FOUND")
            continue

        tmp[col] = build_paths(tmp)
        exists = _check_paths_exist(tmp[col])
        tmp = tmp[exists]

        print(f"[INFO] {col}: {len(tmp)} images found")

        if tmp.empty:
            continue

        tmp = (
            tmp.sort_values(["study_id", "dicom_id"])
               .drop_duplicates("study_id")
        )

        out.append(tmp[["study_id", col]])

    assert out, "NO IMAGES FOUND AT ALL — CHECK PATH LOGIC"

    meta = out[0]
    for t in out[1:]:
        meta = meta.merge(t, on="study_id", how="outer")

    print(f"[INFO] METADATA FEATURES SHAPE: {meta.shape}")
    return meta
