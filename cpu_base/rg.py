import pandas as pd
from pathlib import Path
import requests
import json
from torch.utils.data import Dataset, DataLoader
import time
import yaml
from tqdm import tqdm

class NotesDataset(Dataset):
    def __init__(self, csv_path, notes_base, text_col):
        self.df = pd.read_csv(csv_path)
        self.notes_base = Path(notes_base)
        self.text_col = text_col

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        text_path = self.notes_base / row[self.text_col]
        with open(text_path, "r", encoding="utf-8") as f:
            llm_input = f.read()
        return row["subject_id"], row["study_id"], llm_input

def extract_json(text):
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1:
        return text.strip()
    return text[start:end+1]

def ask_ollama_batch(prompts, retries=3, timeout=60):
    url = "http://localhost:11434/api/generate"
    payloads = [{"model":"llama3.1:70b","prompt":p,"stream":False,"options":{"temperature":0}} for p in prompts]
    results = []
    for payload in payloads:
        last_error = None
        for attempt in range(retries):
            try:
                start_time = time.time()
                response = requests.post(url, json=payload, timeout=timeout)
                duration_seconds = time.time() - start_time
                data = response.json()
                raw_model_output = data.get("response","")
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
                results.append(result)
                break
            except Exception as e:
                last_error = str(e)
        else:
            results.append({"error":"Request failed after retries","details":last_error})
    return results

with open("./config/config.yaml") as f:
    cfg = yaml.safe_load(f)

dataset = NotesDataset(
    cfg["paths"]["mimic_prepared_csv"],
    cfg["paths"]["notes"],
    "path_clinical_note"
)
loader = DataLoader(dataset, batch_size=4, shuffle=False)

results = []
total_batches = len(loader)
start_global = time.time()

for i, batch in enumerate(tqdm(loader, desc="Processing batches")):
    subject_ids, study_ids, texts = zip(*batch)
    prompts = [f"""
Your task is to output ONLY valid JSON that strictly follows the schema below.
If your output does not match the schema, FIX IT and output corrected JSON.

SCHEMA:
{{
    "risk_level": "string",
    "score": "number",
    "issues": ["array of strings"]
}}

INPUT:
{text}
""" for text in texts]
    batch_start = time.time()
    batch_results = ask_ollama_batch(prompts)
    batch_duration = time.time() - batch_start
    elapsed_global = time.time() - start_global
    remaining_est = (elapsed_global / (i+1)) * (total_batches - (i+1))
    tqdm.write(f"Batch {i+1}/{total_batches} processed in {batch_duration:.2f}s, elapsed: {elapsed_global:.2f}s, est remaining: {remaining_est:.2f}s")
    for sid, stid, res in zip(subject_ids, study_ids, batch_results):
        row = {"subject_id": sid.item(), "study_id": stid.item()}
        row.update(res)
        results.append(row)

sc_df = pd.DataFrame(results)
sc_df.to_csv("./results/embedding_sc.csv", index=False)

with open("./results/embedding_sc.json","w",encoding="utf-8") as f:
    json.dump(results,f,ensure_ascii=False)

print("Finished: JSON SC saved to ./results/embedding_sc.csv and embedding_sc.json")
