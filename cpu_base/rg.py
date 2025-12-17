import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from pathlib import Path
import yaml
import json

from src.text.llama_loader import LlamaEncoder

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

with open("./config/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = "cuda" if torch.cuda.is_available() else "cpu"

dataset = NotesDataset(
    cfg["paths"]["mimic_prepared_csv"],
    cfg["paths"]["notes"],
    "path_clinical_note"
)
loader = DataLoader(dataset, batch_size=1, shuffle=False)

llama_model = LlamaEncoder(device=device)
results = []

with torch.no_grad():
    for subject_id, study_id, llm_input in loader:
        embedding = llama_model.encode_text(llm_input)
        embedding = embedding.cpu().numpy().flatten()
        row = {"subject_id": subject_id.item(), "study_id": study_id.item(), "Sc": embedding.tolist()}
        results.append(row)

sc_df = pd.DataFrame([{"subject_id": r["subject_id"], "study_id": r["study_id"], **{f"Sc_{i}": v for i,v in enumerate(r["Sc"])}} for r in results])
sc_df.to_csv("./results/embedding_sc.csv", index=False)

with open("./results/embedding_sc.json", "w", encoding="utf-8") as f:
    json.dump(results, f, ensure_ascii=False)

print("Finished: embeddings saved to ./results/embedding_sc.csv and embedding_sc.json")

