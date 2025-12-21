import pandas as pd
import torch
import timm
from torch.utils.data import DataLoader
from torch import nn, optim
from src.vision.dataset import DualViewCXRDataset
from src.vision.model import DualViewSwin
from src.vision.train import train_swin
from src.vision.utils import load_cfg, cxr_transform

cfg = load_cfg("./config/vision_config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.backends.cudnn.benchmark = True

dataset = DualViewCXRDataset(
    cfg["paths"]["csv"],
    cfg,
    cfg["data"]["train_split"],
    cxr_transform()
)

loader = DataLoader(
    dataset,
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
optimizer = optim.AdamW(
    model.parameters(),
    lr=cfg["training"]["lr"]
)

train_swin(
    model,
    loader,
    optimizer,
    criterion,
    device,
    cfg["training"]["epochs"]
)

torch.save(model.state_dict(), cfg["paths"]["checkpoint"])

# ========== Cm extraction for dual-view ==========
def extract_cm_dual(model, frontal, lateral):
    x_f = model.backbone(frontal)
    x_l = model.backbone(lateral)
    cm = (x_f + x_l) / 2  # یا torch.cat([x_f, x_l], dim=1) اگر خواستی کانکات
    return cm

model.eval()
rows = []

with torch.no_grad():
    for batch in loader:
        frontal = batch[0].to(device)
        lateral = batch[1].to(device)
        subject_id = batch[-2]
        study_id = batch[-1]

        Cm = extract_cm_dual(model, frontal, lateral)

        for i in range(Cm.size(0)):
            row = {
                "subject_id": subject_id[i].item(),
                "study_id": study_id[i].item()
            }
            for j, v in enumerate(Cm[i].cpu().tolist()):
                row[f"cm_{j}"] = v
            rows.append(row)

pd.DataFrame(rows).to_csv(cfg["paths"]["cm_output"], index=False)

print("Finished: Cm embeddings saved.")
