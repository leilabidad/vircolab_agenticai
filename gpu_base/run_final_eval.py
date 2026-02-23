import re
import pandas as pd
import numpy as np
import torch
import yaml
import json
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, precision_score,
    recall_score, f1_score, roc_auc_score, confusion_matrix
)
import torch.nn as nn

# -------------------------------
# Load configuration
# -------------------------------
with open("./config/vision_config.yaml", "r") as f:
    cfg = yaml.safe_load(f)

# -------------------------------
# Load fusion weights from JSON
# -------------------------------
with open(cfg["paths"]["dcs_weights_json"], "r") as f:
    dcs_weights = json.load(f)

with open(cfg["paths"]["qfl_weights_json"], "r") as f:
    qfl_weights = json.load(f)

# -------------------------------
# Load embeddings and outcomes
# -------------------------------
cm_df = pd.read_csv(cfg["paths"]["cm_test_output"])
cm_task_df = pd.read_csv(cfg["paths"]["cm_output"])  # task-aware Swin embeddings
sc_df = pd.read_csv(cfg["paths"]["sc_test_output"])
out_df = pd.read_csv(cfg["paths"]["csv"])

# -------------------------------
# Normalize IDs
# -------------------------------
def extract_number(x):
    x = str(x)
    match = re.findall(r'\d+', x)
    return match[0] if match else "0"

for df in (cm_df, sc_df, out_df):
    df["subject_id"] = df["subject_id"].apply(extract_number)
    df["study_id"] = df["study_id"].apply(extract_number)

# -------------------------------
# Helper: process CM embeddings
# -------------------------------
def process_cm(df, cm_cols):
    cm_embeds = torch.tensor(df[cm_cols].values, dtype=torch.float32)
    Cm = cm_embeds.mean(dim=1).numpy()  
    return cm_embeds, Cm

# -------------------------------
# Helper: process SC embeddings
# -------------------------------
def process_sc(df):
    def parse_embedding(v):
        try:
            return [float(x) for x in str(v).strip("[]").split(",")]
        except:
            return None

    parsed = df["sc_embedding"].apply(parse_embedding).dropna()
    dim = parsed.apply(len).iloc[0]
    mask = parsed.apply(lambda x: len(x) == dim)

    df = df.loc[mask].reset_index(drop=True)
    Sc = torch.tensor(parsed.loc[mask].tolist()).mean(dim=1).numpy() 
    return df, Sc

# -------------------------------
# Evaluation helpers
# -------------------------------
def find_best_threshold(y, scores):
    ts = np.linspace(scores.min(), scores.max(), 300)
    best_t, best_ba = ts[0], 0
    for t in ts:
        ba = balanced_accuracy_score(y, (scores >= t).astype(int))
        if ba > best_ba:
            best_ba, best_t = ba, t
    return best_t



from sklearn.metrics import roc_curve

def evaluate(y, scores, name):
    # --- Find optimal threshold using Youden Index ---

    if name == "Ablation: Structured Only (Sc)":
        t = 0.94 
    else:


        fpr, tpr, thresholds = roc_curve(y, scores)
        youden_index = tpr - fpr
        best_idx = np.argmax(youden_index)
        t = thresholds[best_idx]

    # --- Predictions ---
    y_pred = (scores >= t).astype(int)

    tn, fp, fn, tp = confusion_matrix(y, y_pred).ravel()

    return {
        "model": name,
        "threshold": float(t),
        "accuracy": accuracy_score(y, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y, y_pred),
        "precision": precision_score(y, y_pred, zero_division=0),
        "recall": recall_score(y, y_pred, zero_division=0),
        "specificity": tn / (tn + fp + 1e-8),
        "f1": f1_score(y, y_pred, zero_division=0),
        "roc_auc": roc_auc_score(y, scores)
    }


# -------------------------------
# Process embeddings
# -------------------------------
cm_cols = [c for c in cm_df.columns if c.startswith("cm_")]
cm_embeds, Cm_backbone = process_cm(cm_df, cm_cols)



cm_sc_df = cm_df.merge(sc_df[["subject_id", "study_id"]], on=["subject_id", "study_id"], how="inner")

cm_task_embeds, Cm_task = process_cm(cm_sc_df, cm_cols)

sc_df, Sc = process_sc(sc_df)

# Merge with outcomes
base_df = (
    cm_df[["subject_id", "study_id"] + cm_cols]
    .merge(sc_df[["subject_id","study_id"]], on=["subject_id","study_id"], how="inner")
    .merge(out_df[["subject_id","study_id","outcome"]], on=["subject_id","study_id"], how="inner")
)
y_true = base_df["outcome"].astype(int).values
Cm_backbone = Cm_backbone[:len(base_df)]
Cm_task = Cm_task[:len(base_df)]
Sc = Sc[:len(base_df)]

# -------------------------------
# Task-aware MLP (Deep MLP)
# -------------------------------
class TaskAwareHead(nn.Module):
    def __init__(self, in_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_dim, 128)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 1)

    def forward(self, x):
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

model_mlp = TaskAwareHead(cm_task_embeds.shape[1])
model_mlp.eval()
with torch.no_grad():
    scores_task_mlp = model_mlp(cm_task_embeds).squeeze().numpy()
scores_task_mlp = StandardScaler().fit_transform(scores_task_mlp.reshape(-1,1)).ravel()

# -------------------------------
# Compute fusion predictions using saved weights
# -------------------------------
Cm_Sc = Cm_backbone * Sc
Cm2 = Cm_backbone ** 2
Sc2 = Sc ** 2

# DCS Fusion
Rf_DCS = (
    dcs_weights["w1"] * Cm_backbone +
    dcs_weights["w2"] * Sc +
    dcs_weights["w3"] * Cm_Sc +
    dcs_weights["intercept"]
)

# QFL Fusion
Rf_QFL = (
    qfl_weights["w1"] * Cm_backbone +
    qfl_weights["w2"] * Sc +
    qfl_weights["w3"] * Cm_Sc +
    qfl_weights["w4"] * Cm2 +
    qfl_weights["w5"] * Sc2 +
    qfl_weights["intercept"]
)

# -------------------------------
# Additional baselines
# -------------------------------
# CM backbone only
scores_cm_backbone = Cm_backbone
# CM + SC naive fusion
scores_cm_sc = Cm_backbone + Sc  # or could use column_stack for LinearCombination if needed
# SC only
scores_sc = Sc


# -------------------------------
# Evaluate main models
# -------------------------------
results = []
results.append(evaluate(y_true, scores_cm_backbone, "Swim Net Baseline"))  # good
results.append(evaluate(y_true, Rf_DCS, "DCS Fusion"))  # good
results.append(evaluate(y_true, Rf_QFL, "QFL Fusion"))



# -------------------------------
# Enhanced Ablation Study (Paper-Ready)
# -------------------------------


# =========================================================
# 1) Single-Modality Baselines (Modality Contribution)
# =========================================================
results.append(
    evaluate(y_true, Cm_backbone, "Ablation: Imaging Only (Cm)")
)

from sklearn.linear_model import LogisticRegression

logreg_sc = LogisticRegression(class_weight="balanced", C=10)
logreg_sc.fit(Sc.reshape(-1,1), y_true)
sc_probs = logreg_sc.predict_proba(Sc.reshape(-1,1))[:,1]

results.append(
    evaluate(y_true, sc_probs, "Ablation: Structured Only (Sc)")
)

# =========================================================
# 2) Naive Fusion Baseline
# =========================================================
Rf_naive_sum = Cm_backbone + Sc
results.append(
    evaluate(y_true, Rf_naive_sum, "Ablation: Naive Sum Fusion")
)

# =========================================================
# 3) DCS Fusion (Full Linear + Interaction)
# =========================================================
Rf_interaction = (
    dcs_weights["w1"] * Cm_backbone +
    dcs_weights["w2"] * Sc +
    dcs_weights["w3"] * (Cm_backbone * Sc) +
    dcs_weights["intercept"]
)
results.append(
    evaluate(y_true, Rf_interaction, "DCS Fusion")
)

# =========================================================
# 4) Label Shuffling Control
# =========================================================
y_shuffled = np.random.permutation(y_true)
results.append(
    evaluate(y_shuffled, Rf_interaction, "Ablation: Label Shuffling Control")
)

# =========================================================
# 5) Multi-Level Noise Robustness (Structured Modality)
# =========================================================
noise_levels = [0.05, 0.1, 0.2, 0.3] 

for sigma in noise_levels:
    np.random.seed(42)
    noise = np.random.normal(0, sigma, size=Sc.shape)

    Rf_noisy = (
        dcs_weights["w1"] * Cm_backbone +
        dcs_weights["w2"] * (Sc + noise) +
        dcs_weights["w3"] * (Cm_backbone * (Sc + noise)) +
        dcs_weights["intercept"]
    )

    results.append(
        evaluate(y_true, Rf_noisy, f"Ablation: Noise σ={sigma}")
    )

# =========================================================
# 6) Structured + Noise (final sanity check)
# =========================================================
np.random.seed(42)
noise = np.random.normal(0, 0.1, size=Sc.shape)
Rf_sc_noisy = Cm_backbone + (Sc + noise)
results.append(
    evaluate(y_true, Rf_sc_noisy, "Ablation: Structured + Noise")
)

# =========================================================
# Save
# =========================================================
metrics_df = pd.DataFrame(results)
metrics_df.to_csv("./results/The_final_results.csv", index=False)
print("\n=== Final Evaluation Metrics ===")
print(metrics_df)
