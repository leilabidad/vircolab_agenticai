import cudf
import cupy as cp
from pathlib import Path

def load_note_gpu(rows, base_path):
    rows_cpu = rows.to_pandas()
    result = []
    for _, row in rows_cpu.iterrows():
        subject_id = str(row.subject_id)
        study_id = str(row.study_id)
        note_path = Path(base_path) / f"p{subject_id[:2]}" / f"p{subject_id}" / f"s{study_id}.txt"
        if note_path.exists() and note_path.stat().st_size > 0:
            result.append(note_path)
        else:
            result.append(None)
    return cudf.from_pandas(pd.Series(result))

def build_note_gpu(rows, labels, cfg):
    note_dir = Path(cfg["paths"]["notes"])
    note_dir.mkdir(parents=True, exist_ok=True)
    rows_cpu = rows.to_pandas()
    texts = []

    for _, row in rows_cpu.iterrows():
        note_path = note_dir / f"{row.study_id}.txt"
        positives = [l for l in labels if row[l] == 1]
        if positives:
            text = "Findings:\n" + "\n".join(f"- {p}" for p in positives)
        else:
            text = "Findings:\n- No acute cardiopulmonary abnormality."
        texts.append(text)
        if not note_path.exists():
            note_path.write_text(text)
    return cudf.from_pandas(pd.Series([note_dir / f"{row.study_id}.txt" for _, row in rows_cpu.iterrows()]))
