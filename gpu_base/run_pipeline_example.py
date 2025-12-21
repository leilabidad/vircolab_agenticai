from agentic_ai.pipeline import run_pipeline_batch
import json
from pathlib import Path

frontal_paths = [
    "./agentic_ai/data/frontal_images/fr1.jpg"
]

lateral_paths = [
    "./agentic_ai/data/lateral_images/la1.jpg"
]

notes_dir = Path("./agentic_ai/data/notes")
llm_text_batch = []
for patient_id in ["1"]:
    note_path = notes_dir / f"s{patient_id}.txt"
    with open(note_path, "r", encoding="utf-8") as f:
        llm_text_batch.append(f.read())

with open("../gpu_base/results/dcs_weights.json") as f:
    dcs_weights = json.load(f)

results = run_pipeline_batch(frontal_paths, lateral_paths, llm_text_batch, dcs_weights)

for i, r in enumerate(results):
    print(f"Patient {i+1}:")
    #print("Cm_vector:", r["Cm_vector"])
    print("Sc:", r["Sc"])
    print("Rf:", r["Rf"])
