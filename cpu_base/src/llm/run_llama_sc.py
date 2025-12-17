import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd

from src.text.llama_model import LlamaEncoder
from src.text.dataset import NotesDataset

cfg = {
    "paths": {
        "notes_csv": "./data/notes.csv",
        "sc_output": "./results/embedding_sc.csv"
    },
    "training": {
        "batch_size": 1
    }
}

device = "cpu"
dataset = NotesDataset(cfg["paths"]["notes_csv"])
loader = DataLoader(dataset, batch_size=cfg["training"]["batch_size"], shuffle=False)

llama_model = LlamaEncoder().to(device)
llama_model.eval()

results = []

with torch.no_grad():
    for subject_id, study_id, llm_input in loader:
        embedding = llama_model.encode_text(llm_input)
        embedding = embedding.cpu().numpy().flatten()
        row = [subject_id.item(), study_id.item()] + embedding.tolist()
        results.append(row)

columns = ["subject_id", "study_id"] + [f"Sc_{i}" for i in range(len(embedding))]
sc_df = pd.DataFrame(results, columns=columns)
sc_df.to_csv(cfg["paths"]["sc_output"], index=False)
print(f"Finished: embeddings saved to {cfg['paths']['sc_output']}")
