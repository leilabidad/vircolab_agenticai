from pathlib import Path

def select_images(metadata, base_path):
    def pick(df, views, col):
        tmp = df[df["ViewPosition"].isin(views)].copy()
        tmp["path"] = tmp.apply(
            lambda r: Path(base_path) / f"{r.study_id}" / f"{r.dicom_id}.jpg",
            axis=1
        )
        tmp = tmp.sort_values(["study_id","dicom_id"]).drop_duplicates("study_id")
        return tmp[["study_id","path"]].rename(columns={"path": col})

    fr = pick(metadata, ["PA","AP"], "path_img_fr")
    la = pick(metadata, ["LATERAL"], "path_img_la")
    return fr, la
