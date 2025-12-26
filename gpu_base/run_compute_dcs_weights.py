import pandas as pd
import torch
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import yaml
import json
import time

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
    raise ValueError("Outcome column is required for DCS regression")

cm_cols = [c for c in cm_df.columns if c.startswith("cm_")]
if len(cm_cols) == 0:
    raise ValueError("No cm_ columns found in Cm dataframe")

cm_embeds = torch.tensor(cm_df[cm_cols].values, dtype=torch.float32)
cm_proj = cm_embeds.mean(dim=1).numpy().reshape(-1, 1)

def parse_embedding(s):
    if pd.isna(s):
        return [0.0]
    if isinstance(s, str):
        return [float(x) for x in s.strip("[]").split(",")]
    elif isinstance(s, (list, tuple)):
        return list(s)
    elif isinstance(s, (float, int)):
        return [float(s)]
    else:
        raise ValueError(f"Cannot parse embedding: {s}")

sc_embeds = torch.tensor(sc_df["sc_embedding"].apply(parse_embedding).tolist(), dtype=torch.float32)
sc_proj = sc_embeds.mean(dim=1).numpy().reshape(-1, 1)

cm_scaled = MinMaxScaler().fit_transform(cm_proj)
sc_scaled = MinMaxScaler().fit_transform(sc_proj)

df = cm_df[["subject_id", "study_id"]].copy()
df["Cm_final_scaled"] = cm_scaled
df["score_scaled"] = sc_scaled

df = df.merge(out_df[["subject_id", "study_id", "outcome"]], on=["subject_id", "study_id"], how="inner")

df["Cm_Sc"] = df["Cm_final_scaled"] * df["score_scaled"]

df = df.dropna()
if len(df) < 10:
    raise ValueError("Too few valid samples for DCS regression")

X = df[["Cm_final_scaled", "score_scaled", "Cm_Sc"]]
y = df["outcome"]

reg = LinearRegression()
reg.fit(X, y)

w1, w2, w3 = reg.coef_
intercept = reg.intercept_

df["Rf_pred"] = reg.predict(X)

df.to_csv(cfg["paths"]["rf_output"], index=False)

weights = {
    "w1": float(w1),
    "w2": float(w2),
    "w3": float(w3),
    "intercept": float(intercept)
}

with open(cfg["paths"]["dcs_weights_json"], "w") as f:
    json.dump(weights, f, indent=4)

print("Finished: DCS weights and predictions saved.")
print("Elapsed time (seconds):", round(time.time() - start_time, 2))
