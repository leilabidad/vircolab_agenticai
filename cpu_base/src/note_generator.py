from pathlib import Path

def load_note(row, base_path):
    subject_id = str(row.subject_id)
    study_id = str(row.study_id)

    note_path = Path(base_path) / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"

    if note_path.exists() and note_path.stat().st_size > 0:
        return note_path
    return None



def build_note(row, labels, cfg):
    note_dir = Path(cfg["paths"]["notes"])
    note_dir.mkdir(parents=True, exist_ok=True)
    note_path = note_dir / f"{row.study_id}.txt"

    if cfg["clinical_note"]["mode"] == "real":
        return note_path if note_path.exists() else None

    positives = [l for l in labels if row[l] == 1]
    if positives:
        text = "Findings:\n" + "\n".join(f"- {p}" for p in positives)
    else:
        text = "Findings:\n- No acute cardiopulmonary abnormality."

    if not note_path.exists():
        note_path.write_text(text)

    return note_path
