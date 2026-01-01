import pandas as pd

def build_split_features(cfg):
    return pd.read_csv(
        cfg["paths"]["split"],
        usecols=["study_id", "split"]
    )
