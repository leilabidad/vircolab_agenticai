import yaml
import torch
from src.loader import load_tables
from src.visit_index import build_visit_index
from src.image_selector import select_images
from src.note_model import ClinicalNoteEncoder
from src.dataset_builder import build_dataset
from src.checkpoint import save_checkpoint

with open("config/config.yaml") as f:
    cfg = yaml.safe_load(f)

device = torch.device(cfg["runtime"]["device"])

chexpert, metadata, split = load_tables(cfg)
visits = build_visit_index(chexpert, split)
fr, la = select_images(metadata, cfg["paths"]["images"])

encoder = ClinicalNoteEncoder(
    cfg["model"]["name"],
    device,
    cfg["model"]["max_length"]
)

df, embeddings = build_dataset(visits, metadata, fr, la, encoder, cfg)
save_checkpoint({"table": df, "embeddings": embeddings}, cfg["paths"]["output"])

print("DONE")
print("Visits:", len(df))
