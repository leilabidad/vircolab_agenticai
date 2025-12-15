from pathlib import Path
from .utils import batched, progress
from .checkpoint import save_checkpoint

def build_dataset(visits, metadata, images_fr, images_la, note_encoder, cfg):
    df = visits.merge(images_fr, on="study_id", how="left")
    df = df.merge(images_la, on="study_id", how="left")

    notes = []
    for sid in df["study_id"]:
        p = Path(cfg["paths"]["clinical_notes"]) / f"{sid}.txt"
        notes.append(p.read_text() if p.exists() else None)

    df["note_text"] = notes

    keep = []
    if "image" in cfg["filtering"]["require_any"]:
        keep.append(df["path_img_fr"].notna() | df["path_img_la"].notna())
    if "note" in cfg["filtering"]["require_any"]:
        keep.append(df["note_text"].notna())

    mask = keep[0]
    for k in keep[1:]:
        mask |= k

    df = df[mask].reset_index(drop=True)

    embeddings = []
    texts = df["note_text"].fillna("").tolist()

    for i, batch in enumerate(progress(list(batched(texts, cfg["runtime"]["batch_size"])), cfg["runtime"]["progress_bar"])):
        emb = note_encoder.encode(batch)
        embeddings.append(emb)

        if (i+1) * cfg["runtime"]["batch_size"] % cfg["runtime"]["save_every"] == 0:
            save_checkpoint(embeddings, cfg["paths"]["output"])

    return df, embeddings
