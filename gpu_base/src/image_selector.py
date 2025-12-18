import cudf
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor

def check_paths_exist(paths):
    with ThreadPoolExecutor() as executor:
        return list(executor.map(lambda p: Path(p).exists(), paths))

def select_images(metadata, base_path):
    df_gpu = cudf.from_pandas(metadata)

    def pick_one(df, views, name):
        tmp = df[df["ViewPosition"].isin(views)].copy()
        tmp_cpu = tmp.to_pandas()
        tmp_cpu["path"] = tmp_cpu.apply(
            lambda r: str(Path(base_path) / f"p{str(r.subject_id)[:2]}" / f"p{r.subject_id}" / f"s{r.study_id}" / f"{r.dicom_id}.jpg"),
            axis=1
        )
        exists_mask = check_paths_exist(tmp_cpu["path"])
        tmp_cpu = tmp_cpu[exists_mask]
        tmp_cpu = tmp_cpu.sort_values(["study_id", "dicom_id"]).drop_duplicates("study_id")
        return cudf.from_pandas(tmp_cpu[["study_id", "path"]].rename(columns={"path": name}))

    frontal = pick_one(df_gpu, ["PA", "AP"], "path_img_fr")
    lateral = pick_one(df_gpu, ["LATERAL"], "path_img_la")
    return frontal, lateral
