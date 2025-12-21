import torch

def compute_rf(cm_score, sc_score_dicts, weights, device="cuda"):
    sc_values = [float(d["score"]) for d in sc_score_dicts]
    sc_score = torch.tensor(sc_values, dtype=torch.float32, device=device)
    sc_scaled = (sc_score - sc_score.min()) / (sc_score.max() - sc_score.min() + 1e-8)
    cm_score = cm_score.to(device)
    cm_min = cm_score.min(1, keepdim=True)[0]
    cm_max = cm_score.max(1, keepdim=True)[0]
    cm_scaled = (cm_score - cm_min) / (cm_max - cm_min + 1e-8)
    rf = (
        weights["w1"] * cm_scaled.mean(1)
        + weights["w2"] * sc_scaled
        + weights["w3"] * (cm_scaled.mean(1) * sc_scaled)
        + weights["intercept"]
    )
    return rf
