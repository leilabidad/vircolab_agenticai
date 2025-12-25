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
    payload = {"model": "llama3.1:70b","prompt": prompt,"stream": False,"options": {"temperature": 0}}
    for _ in range(retries):
        try:
            t0 = time.time()
            r = requests.post(url, json=payload, timeout=timeout)
            elapsed = round(time.time() - t0, 2)
            raw = r.json().get("response", "")
            extracted = extract_json(raw)
            result = {"response_time_sec": elapsed}
            if extracted:
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
    payload = {"model": "llama3.1:70b","prompt": text}
    r = requests.post(url, json=payload, timeout=timeout)
    return r.json()["embedding"]

def show_time():
    now = datetime.now()
    print("Date:", now.strftime("%Y-%m-%d"), "Time:", now.strftime("%H:%M:%S"))

few_shot_block = """
Input: Patient has high blood pressure and diabetes.
Output: {"risk_level":"high","score":0.9,"issues":["high blood pressure","diabetes"]}

Input: Patient is healthy with no chronic conditions.
Output: {"risk_level":"low","score":0.1,"issues":[]}
"""

show_time()
with open("./config/config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = NotesDataset(cfg["paths"]["mimic_prepared_csv"], "path_clinical_note")
loader = DataLoader(dataset, batch_size=1, shuffle=False)

output_csv = Path("./results/embedding_sc.csv")
output_json = Path("./results/embedding_sc.json")
output_csv.parent.mkdir(parents=True, exist_ok=True)

# فایل CSV و JSON آماده می‌کنیم
if not output_csv.exists():
    output_csv.touch()
    with open(output_csv, "w", encoding="utf-8") as f:
        f.write("subject_id,study_id,sc_embedding,risk_level,score,issues\n")

if output_json.exists():
    with open(output_json, "r", encoding="utf-8") as f:
        try:
            processed_data = json.load(f)
            processed_set = set((d["subject_id"], d["study_id"]) for d in processed_data)
        except:
            processed_set = set()
else:
    processed_set = set()

start_time = time.time()
batch = []

for subject_id, study_id, text in loader:
    key = (subject_id.item(), study_id.item())
    if key in processed_set:
        continue

    prompt = f"""
You must output ONLY valid JSON following the schema.

SCHEMA:
{{
  "risk_level": "string",
  "score": "number",
  "issues": ["array of strings"]
}}

EXAMPLES:
{few_shot_block}

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
    batch.append(row)
    processed_set.add(key)

    if len(batch) == 10:
        # اضافه کردن batch به CSV مستقیم
        df_batch = pd.DataFrame(batch)
        df_batch.to_csv(output_csv, mode='a', header=False, index=False)
        # آپدیت JSON
        if output_json.exists():
            with open(output_json, "r+", encoding="utf-8") as f:
                data = json.load(f)
                data.extend(batch)
                f.seek(0)
                json.dump(data, f, ensure_ascii=False)
        else:
            with open(output_json, "w", encoding="utf-8") as f:
                json.dump(batch, f, ensure_ascii=False)
        batch.clear()
        print("=== 10 rows processed! ===")
        show_time()

# باقی‌مانده
if batch:
    df_batch = pd.DataFrame(batch)
    df_batch.to_csv(output_csv, mode='a', header=False, index=False)
    if output_json.exists():
        with open(output_json, "r+", encoding="utf-8") as f:
            data = json.load(f)
            data.extend(batch)
            f.seek(0)
            json.dump(data, f, ensure_ascii=False)
    else:
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(batch, f, ensure_ascii=False)
    print(f"=== {len(batch)} remaining rows processed! ===")
    show_time()

elapsed = round(time.time() - start_time, 2)
print(f"Total elapsed time: {elapsed} seconds")
