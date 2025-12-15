import pandas as pd

def load_tables(cfg):
    chexpert = pd.read_csv(cfg["paths"]["chexpert"])
    metadata = pd.read_csv(cfg["paths"]["metadata"])
    split = pd.read_csv(cfg["paths"]["split"])
    return chexpert, metadata, split
