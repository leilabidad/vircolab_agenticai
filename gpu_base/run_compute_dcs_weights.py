import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler
import yaml
import json
import time
import numpy as np
from scipy.optimize import minimize

start_time = time.time()

with open("./config/vision_config.yaml") as f:
    cfg = yaml.safe_load(f)

cm_df = pd.read_csv(cfg["paths"]["cm_output"])
sc_df = pd.read_csv(cfg["paths"]["sc_output"])
out_df = pd.read_csv(cfg["paths"]["csv"])

for df in (cm_df, sc_df, out_df):
    df["subject_id"] = df["subject_id"].astype(str)
    df["study_id"] = df["study_id"].astype(str)

if "outcome" not in out_df.columns:
    raise ValueError("Outcome column is required")

cm_cols = [c for c in cm_df.columns if c.startswith("cm_")]
if len(cm_cols) == 0:
    raise ValueError("No cm_ columns found")

cm_embeds = torch.tensor(cm_df[cm_cols].values, dtype=torch.float32)
cm_proj = cm_embeds.mean(dim=1).numpy().reshape(-1, 1)

def parse_embedding(s):
    if pd.isna(s):
        return [0.0]
    if isinstance(s, str):
        return [float(x) for x in s.strip("[]").split(",")]
    if isinstance(s, (list, tuple)):
        return list(s)
    return [float(s)]

sc_embeds = torch.tensor(sc_df["sc_embedding"].apply(parse_embedding).tolist(), dtype=torch.float32)
sc_proj = sc_embeds.mean(dim=1).numpy().reshape(-1, 1)

cm_scaled = MinMaxScaler().fit_transform(cm_proj)
sc_scaled = MinMaxScaler().fit_transform(sc_proj)

df = cm_df[["subject_id", "study_id"]].copy()
df["Cm"] = cm_scaled
df["Sc"] = sc_scaled

df = df.merge(out_df[["subject_id", "study_id", "outcome"]], on=["subject_id", "study_id"], how="inner")
df["Cm_Sc"] = df["Cm"] * df["Sc"]
df = df.dropna()

if len(df) < 10:
    raise ValueError("Too few samples")

X = df[["Cm", "Sc", "Cm_Sc"]].values
y = df["outcome"].values

def loss(w):
    pred = X @ w[:3] + w[3]
    return np.mean((pred - y) ** 2)

bounds = [(0, None), (0, None), (0, None), (None, None)]
init = np.array([0.1, 0.1, 0.1, 0.0])

res = minimize(loss, init, bounds=bounds)

w1, w2, w3, intercept = res.x

df["Rf_pred"] = X @ np.array([w1, w2, w3]) + intercept
df.to_csv(cfg["paths"]["rf_output"], index=False)

weights = {
    "w1": float(w1),
    "w2": float(w2),
    "w3": float(w3),
    "intercept": float(intercept)
}

with open(cfg["paths"]["dcs_weights_json"], "w") as f:
    json.dump(weights, f, indent=4)

print("Finished")
print("Elapsed:", round(time.time() - start_time, 2))
