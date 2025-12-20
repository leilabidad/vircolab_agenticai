import pandas as pd
from pathlib import Path
import requests
import json
from torch.utils.data import Dataset, DataLoader
import time
import yaml
from datetime import datetime

class NotesDataset(Dataset):
    def __init__(self, csv_path, notes_base, text_col):
        self.df = pd.read_csv(csv_path)
        self.notes_base = Path(notes_base)
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_path = row[self.text_col]
        with open(text_path, "r", encoding="utf-8") as f:
            llm_input = f.read()
        return row["subject_id"], row["study_id"], llm_input

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text.strip()
    return text[start:end+1]

def ask_ollama(prompt, retries=3, timeout=30):
    url = "http://localhost:11434/api/generate"
    payload = {
        "model": "llama3.1:70b",
        "prompt": prompt,
        "stream": False,
        "options": {"temperature": 0}
    }
    last_error = None
    for attempt in range(retries):
        try:
            start_time = time.time()
            response = requests.post(url, json=payload, timeout=timeout)
            duration_seconds = time.time() - start_time
            data = response.json()
            raw_model_output = data.get("response", "")
            extracted = extract_json(raw_model_output)
            result = {"response_time_seconds": round(duration_seconds,2)}
            try:
                parsed = json.loads(extracted)
                if isinstance(parsed, dict):
                    result.update(parsed)
                else:
                    result["raw_response"] = parsed
            except:
                result["error"] = "Invalid JSON from model"
                result["raw_response"] = extracted
            return result
        except Exception as e:
            last_error = str(e)
    return {"error": "Request failed after retries", "details": last_error}

def showTime():

    now = datetime.now()
    current_time = now.strftime("%H:%M:%S")
    current_date = now.strftime("%Y-%m-%d")

    print("Time:", current_time)
    print("Date:", current_date)


showTime()

with open("./config/config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = NotesDataset(
    cfg["paths"]["mimic_prepared_csv"],
    cfg["paths"]["notes"],
    "path_clinical_note"
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

results = []
for subject_id, study_id, llm_input in loader:
    question = f"""
Your task is to output ONLY valid JSON that strictly follows the schema below.
If your output does not match the schema, FIX IT and output corrected JSON.

SCHEMA:
{{
    "risk_level": "string",
    "score": "number",
    "issues": ["array of strings"]
}}

INPUT:
{llm_input[0]}
"""
    res = ask_ollama(question)
    row = {"subject_id": subject_id.item(), "study_id": study_id.item()}
    row.update(res)
    results.append(row)

sc_df = pd.DataFrame(results)
sc_df.to_csv("./results/embedding_sc.csv", index=False)

with open("./results/embedding_sc.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print("Finished: JSON SC saved to ./results/embedding_sc.csv and embedding_sc.json")

showTime()
