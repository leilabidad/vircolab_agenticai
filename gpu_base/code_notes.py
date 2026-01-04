import pandas as pd
from pathlib import Path
import re
import sys

def count_notes_recursive(csv_path, notes_base_path):
    # Load study_ids from CSV
    df = pd.read_csv(csv_path, usecols=["study_id"])
    study_ids = set(df["study_id"].astype(str).str.strip())

    # Make sure base path exists
    notes_base_path = Path(notes_base_path)
    if not notes_base_path.exists():
        raise FileNotFoundError(f"Notes path not found: {notes_base_path}")

    found = set()
    total_txt = 0

    # Recursively scan all .txt files
    for p in notes_base_path.rglob("*.txt"):
        total_txt += 1
        if p.stat().st_size == 0:
            continue  # ignore empty files

        # extract ALL numbers from filename
        nums = re.findall(r"\d+", p.stem)
        for sid in nums:
            if sid in study_ids:
                found.add(sid)
                break  # only count one match per file

    missing = study_ids - found

    print("===================================")
    print(f"TXT files scanned        : {total_txt}")
    print(f"Unique study_id in CSV   : {len(study_ids)}")
    print(f"Notes found              : {len(found)}")
    print(f"Missing notes            : {len(missing)}")
    print("===================================")

    return found, missing


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python count_notes.py <csv_path> <notes_base_path>")
        sys.exit(1)

    csv_path = sys.argv[1]
    notes_base_path = sys.argv[2]
    count_notes_recursive(csv_path, notes_base_path)
