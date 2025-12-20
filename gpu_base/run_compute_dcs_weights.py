import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import MinMaxScaler
import yaml
import time
import json

start_time = time.time()

with open("./config/vision_config.yaml") as f:
    cfg = yaml.safe_load(f)

sc_df = pd.read_csv(cfg["paths"]["sc_output"])
cm_df = pd.read_csv(cfg["paths"]["cm_output"])
out_df = pd.read_csv(cfg["paths"]["csv"])

for df in (sc_df, cm_df, out_df):
    df["subject_id"] = df["subject_id"].astype(str)
    df["study_id"] = df["study_id"].astype(str)

if "outcome" not in out_df.columns:
    raise ValueError("outcome column is required for DCS regression")

cm_cols = [c for c in cm_df.columns if c.startswith("Cm_")]
if len(cm_cols) == 0:
    raise ValueError("No Cm_ columns found")

cm_df["Cm_final"] = cm_df[cm_cols].mean(axis=1)

cm_df["Cm_final_scaled"] = MinMaxScaler().fit_transform(cm_df[["Cm_final"]])
sc_df["score_scaled"] = MinMaxScaler().fit_transform(sc_df[["score"]])

df = (
    cm_df[["subject_id", "study_id", "Cm_final_scaled"]]
    .merge(sc_df[["subject_id", "study_id", "score_scaled"]], on=["subject_id", "study_id"], how="inner")
    .merge(out_df[["subject_id", "study_id", "outcome"]], on=["subject_id", "study_id"], how="inner")
)

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

df["Rf_pred"] = (
    w1 * df["Cm_final_scaled"]
    + w2 * df["score_scaled"]
    + w3 * df["Cm_Sc"]
    + intercept
)

df.to_csv(cfg["paths"]["rf_output"], index=False)

weights = {
    "w1": float(w1),
    "w2": float(w2),
    "w3": float(w3),
    "intercept": float(intercept),
}

with open(cfg["paths"]["dcs_weights_json"], "w") as f:
    json.dump(weights, f, indent=4)

print("Elapsed time (seconds):", round(time.time() - start_time, 2))
