from pathlib import Path

def load_note_gpu(df, base_path):
    result = []
    for _, row in df.iterrows():
        subject_id = str(row.subject_id)
        study_id = str(row.study_id)
        note_path = Path(base_path) / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"
        if note_path.exists() and note_path.stat().st_size > 0:
            result.append(str(note_path))
        else:
            result.append(None)
    return result

def build_note_gpu(df, labels, cfg):
    note_dir = Path(cfg["paths"]["notes"])
    note_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for _, row in df.iterrows():
        note_path = note_dir / f"{row.study_id}.txt"
        positives = [l for l in labels if row[l] == 1]
        text = "Findings:\n" + "\n".join(f"- {p}" for p in positives) if positives else "Findings:\n- No acute cardiopulmonary abnormality."
        if not note_path.exists():
            note_path.write_text(text)
        paths.append(str(note_path))
    return paths
