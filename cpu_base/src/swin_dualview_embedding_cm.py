import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import pandas as pd
import timm
import yaml

with open('cpu_base/config/config.yaml') as f:
    cfg = yaml.safe_load(f)

dataset_csv = cfg["paths"]["mimic_prepared_csv"]
save_path = cfg["paths"]["embedding_cm_csv"]

class MIMICDualViewDataset(Dataset):
    def __init__(self, csv_file, transform=None):
        self.df = pd.read_csv(csv_file)
        self.df = self.df[self.df['split'] == 'train'].reset_index(drop=True)
        self.transform = transform
    def __len__(self):
        return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_fr = Image.open(row['path_img_fr']).convert('RGB')
        img_la = Image.open(row['path_img_la']).convert('RGB')
        if self.transform:
            img_fr = self.transform(img_fr)
            img_la = self.transform(img_la)
        return row['subject_id'], row['study_id'], img_fr, img_la

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

dataset = MIMICDualViewDataset(dataset_csv, transform=transform)
dataloader = DataLoader(dataset, batch_size=1, shuffle=False)

device = "cpu"

backbone = timm.create_model("swin_large_patch4_window12_384", pretrained=True)
backbone.head = nn.Identity()

class DualViewSwin(nn.Module):
    def __init__(self, backbone):
        super().__init__()
        self.backbone = backbone
    def forward(self, x_fr, x_la):
        emb_fr = self.backbone(x_fr)
        emb_la = self.backbone(x_la)
        return (emb_fr + emb_la) / 2

model = DualViewSwin(backbone).to(device)
model.eval()

results = []
with torch.no_grad():
    for subject_id, study_id, img_fr, img_la in dataloader:
        img_fr, img_la = img_fr.to(device), img_la.to(device)
        emb = model(img_fr, img_la).cpu().numpy().flatten()
        row = [subject_id.item(), study_id.item()] + emb.tolist()
        results.append(row)

columns = ["subject_id", "study_id"] + [f"Cm_{i}" for i in range(emb.shape[0])]
df_out = pd.DataFrame(results, columns=columns)
df_out.to_csv(save_path, index=False)
print(f"Finished: embeddings saved to {save_path}")
