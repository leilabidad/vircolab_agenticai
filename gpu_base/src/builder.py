import cudf
from .note_generator import build_note, load_note
from .utils import progress

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","Pleural Effusion","Pleural Other",
    "Pneumonia","Pneumothorax","Support Devices"
]

def build_dataset(chexpert, split, frontal, lateral, cfg):
    chexpert_gpu = cudf.from_pandas(chexpert)
    split_gpu = cudf.from_pandas(split)
    df = chexpert_gpu[["subject_id","study_id"] + LABELS]
    df = df.merge(split_gpu[["study_id","split"]], on="study_id", how="left")
    df = df.merge(frontal, on="study_id", how="left")
    df = df.merge(lateral, on="study_id", how="left")
    notes = []
    df_cpu = df.to_pandas()
    for _, row in progress(df_cpu.iterrows(), cfg["runtime"]["progress_bar"]):
        if cfg["clinical_note"]["mode"] == "fake":
            notes.append(build_note(row, LABELS, cfg))
        else:
            notes.append(load_note(row, cfg["paths"]["notes"]))
    df_cpu["path_clinical_note"] = notes
    df_cpu["outcome"] = df_cpu[LABELS].eq(1).any(axis=1).astype(int)
    if not cfg["filtering"]["keep_if_missing_all"]:
        cols = ["path_clinical_note", "path_img_fr", "path_img_la"]
        mask = (
            df_cpu[cols].notna().all(axis=1) &
            df_cpu[cols].apply(lambda c: c.astype(str).str.strip() != "").all(axis=1)
        )
        df_cpu = df_cpu[mask]
    return cudf.from_pandas(df_cpu.drop_duplicates("study_id"))
