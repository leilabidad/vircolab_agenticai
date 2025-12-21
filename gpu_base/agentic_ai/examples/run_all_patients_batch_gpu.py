import pandas as pd
import torch
from agentic_ai.pipeline import run_pipeline_batch
import json
from pathlib import Path
import yaml

with open("./agentic_ai/config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["paths"]["csv"])
results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
batch_size = cfg["data"].get("batch_size", 4)

for start_idx in range(0, len(df), batch_size):
    end_idx = min(start_idx + batch_size, len(df))
    batch = df.iloc[start_idx:end_idx]

    frontal_batch = torch.stack([torch.load(Path(row["frontal_image"]), map_location=device) for _, row in batch.iterrows()])
    lateral_batch = torch.stack([torch.load(Path(row["lateral_image"]), map_location=device) for _, row in batch.iterrows()])
    llm_text_batch = [Path(row["clinical_note"]).read_text(encoding="utf-8") for _, row in batch.iterrows()]

    dcs_weights = cfg["paths"]["dcs_weights"]
    batch_results = run_pipeline_batch(frontal_batch, lateral_batch, llm_text_batch, dcs_weights)
    for i, (_, row) in enumerate(batch.iterrows()):
        results.append({
            "subject_id": row["subject_id"],
            "study_id": row["study_id"],
            "Cm_vector": batch_results[i]["Cm_vector"],
            "Sc": batch_results[i]["Sc"],
            "Rf": batch_results[i]["Rf"]
        })

pd.DataFrame(results).to_csv("./results/patient_results_batch_gpu.csv", index=False)
with open("./results/patient_results_batch_gpu.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)



