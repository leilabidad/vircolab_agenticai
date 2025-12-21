import pandas as pd
from pathlib import Path
import requests
import json
from torch.utils.data import Dataset, DataLoader
import time
import yaml
from datetime import datetime


class NotesDataset(Dataset):
    def __init__(self, csv_path, text_col):
        self.df = pd.read_csv(csv_path)
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        with open(row[self.text_col], "r", encoding="utf-8") as f:
            text = f.read()
        return row["subject_id"], row["study_id"], text


def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return None
    return text[start:end + 1]


def ask_ollama(prompt, retries=3, timeout=60):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1:70b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }

    for _ in range(retries):
        try:
            t0 = time.time()
            r = requests.post(url, json=payload, timeout=timeout)
            elapsed = round(time.time() - t0, 2)
            raw = r.json().get("response", "")
            extracted = extract_json(raw)

            result = {"response_time_sec": elapsed}

            if extracted is not None:
                try:
                    parsed = json.loads(extracted)
                    result.update(parsed)
                except:
                    result["error"] = "Invalid JSON"
                    result["raw_response"] = extracted
            else:
                result["error"] = "No JSON found"
                result["raw_response"] = raw

            return result
        except Exception as e:
            last_error = str(e)

    return {"error": "LLM request failed", "details": last_error}


def get_llm_embedding(text, timeout=60):
    url = "http://localhost:11434/api/embeddings"
    payload = {
        "model": "llama3.1:70b",
        "prompt": text
    }
    r = requests.post(url, json=payload, timeout=timeout)
    return r.json()["embedding"]


def show_time():
    now = datetime.now()
    print("Date:", now.strftime("%Y-%m-%d"),
          "Time:", now.strftime("%H:%M:%S"))


show_time()

with open("./config/config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = NotesDataset(
    cfg["paths"]["mimic_prepared_csv"],
    "path_clinical_note"
)

loader = DataLoader(dataset, batch_size=1, shuffle=False)

results = []

for subject_id, study_id, text in loader:
    prompt = f"""
Output ONLY valid JSON following the schema below.

SCHEMA:
{{
  "risk_level": "string",
  "score": "number",
  "issues": ["array of strings"]
}}

INPUT:
{text[0]}
"""

    sc_struct = ask_ollama(prompt)
    sc_embed = get_llm_embedding(text[0])

    row = {
        "subject_id": subject_id.item(),
        "study_id": study_id.item(),
        "sc_embedding": sc_embed
    }
    row.update(sc_struct)
    results.append(row)

pd.DataFrame(results).to_csv("./results/embedding_sc.csv", index=False)

with open("./results/embedding_sc.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print("Finished: Sc embeddings saved.")
show_time()
