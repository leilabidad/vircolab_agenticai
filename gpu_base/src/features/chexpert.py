import pandas as pd

LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","Pleural Effusion","Pleural Other",
    "Pneumonia","Pneumothorax","Support Devices"
]

def build_chexpert_features(cfg):
    return pd.read_csv(
        cfg["paths"]["chexpert"],
        usecols=["subject_id", "study_id"] + LABELS
    )
