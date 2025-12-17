import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


class DualViewCXRDataset(Dataset):
    def __init__(self, csv_path, cfg, split, transform=None):
        self.cfg = cfg
        self.df = pd.read_csv(csv_path)
        self.df = self.df[
            self.df[cfg["data"]["split_column"]] == split
        ].reset_index(drop=True)

        self.transform = transform
        self.fr_col = cfg["data"]["image_columns"]["frontal"]
        self.la_col = cfg["data"]["image_columns"]["lateral"]
        self.label_cols = cfg["data"]["label_columns"]

        self.sid = cfg["data"]["id_columns"]["subject_id"]
        self.stid = cfg["data"]["id_columns"]["study_id"]

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_fr = Image.open(row[self.fr_col]).convert("RGB")
        img_la = Image.open(row[self.la_col]).convert("RGB")

        if self.transform:
            img_fr = self.transform(img_fr)
            img_la = self.transform(img_la)

        y = torch.tensor(
            row[self.label_cols].values,
            dtype=torch.float32
        )

        return img_fr, img_la, y, row[self.sid], row[self.stid]
