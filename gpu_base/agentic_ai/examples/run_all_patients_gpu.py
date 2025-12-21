import pandas as pd
import torch
from agentic_ai.pipeline import run_pipeline
import json
from pathlib import Path
import yaml

with open("./agentic_ai/config.yaml") as f:
    cfg = yaml.safe_load(f)

df = pd.read_csv(cfg["paths"]["csv"])
results = []

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

for idx, row in df.iterrows():
    frontal_path = Path(row["frontal_image"])
    lateral_path = Path(row["lateral_image"])
    llm_text_path = Path(row["clinical_note"])

    frontal_img = torch.load(frontal_path, map_location=device)
    lateral_img = torch.load(lateral_path, map_location=device)

    with open(llm_text_path, "r", encoding="utf-8") as f:
        llm_text = f.read()

    dcs_weights = cfg["paths"]["dcs_weights"]
    res = run_pipeline(frontal_img, lateral_img, llm_text, dcs_weights)
    results.append({
        "subject_id": row["subject_id"],
        "study_id": row["study_id"],
        "Cm_vector": res["Cm_vector"],
        "Sc": res["Sc"],
        "Rf": res["Rf"]
    })

pd.DataFrame(results).to_csv("./results/patient_results_gpu.csv", index=False)
with open("./results/patient_results_gpu.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)
