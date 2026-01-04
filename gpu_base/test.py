import pandas as pd
from pathlib import Path
import sys

def count_notes_recursive(csv_path, notes_base_path):
    df = pd.read_csv(csv_path, usecols=["study_id"])
    df["study_id"] = df["study_id"].astype(str).str.strip()

    study_ids = set(df["study_id"].unique())
    found = set()

    notes_base_path = Path(notes_base_path)
    assert notes_base_path.exists(), f"Notes path not found: {notes_base_path}"

    for p in notes_base_path.rglob("*.txt"):
        name = p.stem  # e.g. s123456
        if not name.startswith("s"):
            continue

        sid = name[1:]
        if sid in study_ids and p.stat().st_size > 0:
            found.add(sid)

    print("===================================")
    print(f"Total unique study_id in CSV : {len(study_ids)}")
    print(f"Notes found on disk          : {len(found)}")
    print(f"Missing notes                : {len(study_ids - found)}")
    print("===================================")

    return found, study_ids - found


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage:")
        print("  python count_notes.py <csv_path> <notes_base_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    notes_base_path = sys.argv[2]

    count_notes_recursive(csv_path, notes_base_path)
