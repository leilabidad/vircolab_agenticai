import pandas as pd
import os
import yaml

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

csv_path = cfg["paths"]["output"]

df = pd.read_csv(csv_path)

required_columns = {"outcome", "path_clinical_note", "subject_id"}
missing = required_columns - set(df.columns)
if missing:
    raise ValueError(f"Missing required columns: {missing}")

initial_total = len(df)

def file_invalid(path):
    if not isinstance(path, str):
        return True
    if not os.path.exists(path):
        return True
    return os.path.getsize(path) == 0

# -------- Step 1: Remove invalid clinical notes with outcome=0 --------
mask_invalid_zero = (df["outcome"] == 0) & (df["path_clinical_note"].apply(file_invalid))
df = df[~mask_invalid_zero].reset_index(drop=True)

# -------- Step 2: Balance ratio (patient-aware, ONLY outcome=0) --------
count_one = (df["outcome"] == 1).sum()
count_zero = (df["outcome"] == 0).sum()

target_zero = int((count_one * 7) / 3)

zeros_to_remove = max(0, count_zero - target_zero)

if zeros_to_remove > 0:
    zero_df = df[df["outcome"] == 0]

    zero_indices = zero_df.index.tolist()[::-1]

    removed_indices = []
    used_subjects = set()

    for idx in zero_indices:
        if len(removed_indices) >= zeros_to_remove:
            break

        subject = df.loc[idx, "subject_id"]
        if subject in used_subjects:
            continue

        removed_indices.append(idx)
        used_subjects.add(subject)

    df = df.drop(index=removed_indices).reset_index(drop=True)

# -------- Save result --------
df.to_csv(csv_path, index=False)

# -------- Final statistics --------
final_total = len(df)
final_zero = (df["outcome"] == 0).sum()
final_one = (df["outcome"] == 1).sum()
total_removed = initial_total - final_total
unique_patients = df["subject_id"].nunique()

print(f"Total rows removed: {total_removed}")
print(f"Total rows remaining: {final_total}")
print(f"Outcome=0 rows: {final_zero}")
print(f"Outcome=1 rows: {final_one}")
print(f"Unique patients: {unique_patients}")
