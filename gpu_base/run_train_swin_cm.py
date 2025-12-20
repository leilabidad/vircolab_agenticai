import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from torch import nn, optim
from src.vision.dataset import DualViewCXRDataset
from src.vision.model import DualViewSwin
from src.vision.train import train_swin
from src.vision.extract import extract_cm
from src.vision.utils import load_cfg, cxr_transform

cfg = load_cfg("./config/vision_config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

ds = DualViewCXRDataset(
    cfg["paths"]["csv"],
    cfg,
    cfg["data"]["train_split"],
    cxr_transform()
)

loader = DataLoader(
    ds,
    batch_size=cfg["training"]["batch_size"],
    shuffle=True
)

backbone = timm.create_model(
    "swin_base_patch4_window7_224",
    pretrained=True,
    num_classes=0
)

model = DualViewSwin(
    backbone,
    out_dim=len(cfg["data"]["label_columns"])
).to(device)

criterion = nn.BCEWithLogitsLoss()
optimizer = optim.AdamW(model.parameters(), lr=cfg["training"]["lr"])

train_swin(
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg["training"]["epochs"]
)

torch.save(model.state_dict(), cfg["paths"]["checkpoint"])

extract_cm(model, loader, device, cfg["paths"]["cm_output"])

cm_df = pd.read_csv(cfg["paths"]["cm_output"])

for col in cm_df.columns:
    if col in ["subject_id","study_id"]:
        cm_df[col] = cm_df[col].astype(str).str.replace("tensor","").str.replace("(","").str.replace(")","").str.strip().astype(int)
    else:
        cm_df[col] = cm_df[col].astype(str).str.replace("tensor","").str.replace("[","").str.replace("]","").str.replace("\n","").str.strip().astype(float)

cm_df.to_csv(cfg["paths"]["cm_output"], index=False)
