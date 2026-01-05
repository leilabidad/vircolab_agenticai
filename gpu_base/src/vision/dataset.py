import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib

class DualViewCXRDataset(Dataset):
    def __init__(self, csv_path, cfg, split, transform=None):
        self.cfg = cfg
        self.df = pd.read_csv(csv_path)

        self._ensure_stable_split(split)

        self.df = self.df[self.df[cfg["data"]["split_column"]] == split].reset_index(drop=True)

        self.transform = transform
        self.fr_col = cfg["data"]["image_columns"]["frontal"]
        self.la_col = cfg["data"]["image_columns"]["lateral"]
        self.label_cols = cfg["data"]["label_columns"]

        self.sid = cfg["data"]["id_columns"]["subject_id"]
        self.stid = cfg["data"]["id_columns"]["study_id"]

    def _ensure_stable_split(self, split):
        col = self.cfg["data"]["split_column"]
        if col not in self.df.columns:
            # Stable hash with modulo to prevent overflow
            def stable_hash(x):
                return int(hashlib.md5(str(x).encode()).hexdigest(), 16) % (2**32)
            
            hashes = self.df.index.map(stable_hash)
            sorted_idx = self.df.index[np.argsort(hashes)]

            n = len(sorted_idx)
            n_train = int(n * 0.7)
            n_val = int(n * 0.15)

            self.df[col] = ""
            self.df.loc[sorted_idx[:n_train], col] = "train"
            self.df.loc[sorted_idx[n_train:n_train+n_val], col] = "val"
            self.df.loc[sorted_idx[n_train+n_val:], col] = "test"
        # If column exists, leave it unchanged to preserve original behavior

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_fr = Image.open(row[self.fr_col]).convert("RGB")
        img_la = Image.open(row[self.la_col]).convert("RGB")

        if self.transform:
            img_fr = self.transform(img_fr)
            img_la = self.transform(img_la)

        labels = row[self.label_cols].values.astype(np.float32)
        labels = np.nan_to_num(labels, nan=0.0)
        y = torch.tensor(labels, dtype=torch.float32)

        return img_fr, img_la, y, row[self.sid], row[self.stid]
