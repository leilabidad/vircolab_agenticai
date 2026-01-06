import pandas as pd
from pathlib import Path
import requests
import json
from torch.utils.data import Dataset, DataLoader
import time
import yaml
from datetime import datetime
import math
from typing import Dict, Any, List, Optional

# =========================
# Explicit semantic mapping for structured fields
# =========================
MIMIC_FIELD_SEMANTICS = {
    "Atelectasis": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Cardiomegaly": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Consolidation": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Edema": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Enlarged Cardiomediastinum": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Fracture": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Lung Lesion": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Lung Opacity": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Pleural Effusion": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Pleural Other": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Pneumonia": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Pneumothorax": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "Support Devices": {-1.0: "Unknown", 0.0: "Absent", 1.0: "Present"},
    "gender": {0.0: "Male", 1.0: "Female"}
}

# =========================
# Dataset
# =========================
class NotesDataset(Dataset):
    def __init__(self, csv_path: str, text_col: str):
        self.df = pd.read_csv(csv_path)
        self.text_col = text_col

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        row = self.df.iloc[idx].to_dict()

        note_text = ""
        try:
            with open(row[self.text_col], "r", encoding="utf-8") as f:
                note_text = f.read().strip()
        except Exception:
            pass

        row["clinical_note_text"] = note_text if note_text else "MISSING_NOTE"
        return row

def collate_fn(batch: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    return batch

# =========================
# Utilities
# =========================
def safe_value(val: Any) -> Optional[float]:
    if val is None:
        return None
    if isinstance(val, float) and math.isnan(val):
        return None
    return val

def format_structured_block(row: Dict[str, Any]) -> str:
    lines = []
    for field, mapping in MIMIC_FIELD_SEMANTICS.items():
        raw_val = safe_value(row.get(field))
        if raw_val is None:
            semantic = "Missing"
        else:
            semantic = mapping.get(raw_val, "Unknown")
        lines.append(f"{field}: {semantic}")
    return "\n".join(lines)

def build_fusion_text(row: Dict[str, Any]) -> str:
    structured_block = format_structured_block(row)
    clinical_note = row.get("clinical_note_text", "MISSING_NOTE")

    return f"""
[STRUCTURED CLINICAL VARIABLES]
The following fields are machine-extracted, incomplete, and may be noisy.
They must NOT be treated as ground truth.

{structured_block}

[UNSTRUCTURED CLINICAL NOTE]
{clinical_note}
""".strip()

def extract_json(text: str) -> Optional[str]:
    if not isinstance(text, str):
        return None
    s = text.find("{")
    e = text.rfind("}")
    if s == -1 or e == -1 or e <= s:
        return None
    return text[s:e + 1]

def validate_schema(obj: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    if not isinstance(obj, dict):
        return None

    if obj.get("risk_level") not in {"low", "medium", "high"}:
        return None

    try:
        score = float(obj.get("score"))
    except Exception:
        return None

    issues = obj.get("issues")
    if not isinstance(issues, list):
        return None

    return {
        "risk_level": obj["risk_level"],
        "score": score,
        "issues": [str(i) for i in issues]
    }

def ask_ollama(prompt: str, retries: int = 3, timeout: int = 60) -> Dict[str, Any]:
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1:70b",
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": 0,
            "top_p": 1
        }
    }

    last_error = None

    for _ in range(retries):
        try:
            t0 = time.time()
            r = requests.post(url, json=payload, timeout=timeout)
            elapsed = round(time.time() - t0, 2)

            raw = r.json().get("response", "")
            extracted = extract_json(raw)
            if not extracted:
                last_error = "No JSON found"
                continue

            parsed = json.loads(extracted)
            validated = validate_schema(parsed)
            if validated:
                validated["response_time_sec"] = elapsed
                return validated

            last_error = "Schema validation failed"

        except Exception as e:
            last_error = str(e)

    return {
        "risk_level": None,
        "score": None,
        "issues": [],
        "response_time_sec": None,
        "error": last_error
    }

def get_llm_embedding(text: str, timeout: int = 60) -> List[float]:
    url = "http://localhost:11434/api/embeddings"
    payload = {"model": "llama3.1:70b", "prompt": text}
    r = requests.post(url, json=payload, timeout=timeout)
    return r.json().get("embedding", [])

def show_time() -> None:
    print(datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# =========================
# Prompt (Leakage-aware, calibration-aware)
# =========================
PROMPT_TEMPLATE = """
You are a clinical risk assessment system.

Rules:
- Output ONLY valid JSON.
- Do NOT repeat input text.
- Structured variables may be incomplete or incorrect.
- Base your decision primarily on the clinical note.
- Use structured data only as weak contextual hints.

Schema:
{{
  "risk_level": "low | medium | high",
  "score": number between 0 and 1,
  "issues": array of short strings
}}

Input:
{fusion_text}
"""

# =========================
# Main
# =========================
with open("./config/config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = NotesDataset(cfg["paths"]["mimic_prepared_csv"], "path_clinical_note")
loader = DataLoader(dataset, batch_size=1, shuffle=False, collate_fn=collate_fn)

output_csv = Path("./results/embedding_sc.csv")
output_csv.parent.mkdir(parents=True, exist_ok=True)

if not output_csv.exists():
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("subject_id,study_id,sc_embedding,risk_level,score,issues,response_time_sec\n")

allowed_df = pd.read_csv(cfg["paths"]["embedding_cm_csv"])
allowed_ids = set(allowed_df["study_id"].unique())
allowed_rows_count = len(allowed_ids)

processed = set()
batch = []
start_time = time.time()

for rows in loader:
    row = rows[0]

    if row["study_id"] not in allowed_ids:
        continue

    key = (row["subject_id"], row["study_id"])
    if key in processed:
        continue

    fusion_text = build_fusion_text(row)
    prompt = PROMPT_TEMPLATE.format(fusion_text=fusion_text)

    sc_struct = ask_ollama(prompt)
    sc_embedding = get_llm_embedding(fusion_text)

    output_row = {
        "subject_id": row["subject_id"],
        "study_id": row["study_id"],
        "sc_embedding": json.dumps(sc_embedding),
        "risk_level": sc_struct.get("risk_level"),
        "score": sc_struct.get("score"),
        "issues": json.dumps(sc_struct.get("issues", [])),
        "response_time_sec": sc_struct.get("response_time_sec")
    }

    batch.append(output_row)
    processed.add(key)

    if len(batch) == 10:
        pd.DataFrame(batch).to_csv(output_csv, mode="a", header=False, index=False)
        batch.clear()
        show_time()
        print(f"Processed: {len(processed)} | Remaining approx: {allowed_rows_count - len(processed)}")

if batch:
    pd.DataFrame(batch).to_csv(output_csv, mode="a", header=False, index=False)
    show_time()

print("Elapsed seconds:", round(time.time() - start_time, 2))
