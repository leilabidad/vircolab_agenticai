LABELS = [
    "Atelectasis","Cardiomegaly","Consolidation","Edema",
    "Enlarged Cardiomediastinum","Fracture","Lung Lesion",
    "Lung Opacity","Pleural Effusion","Pleural Other",
    "Pneumonia","Pneumothorax","Support Devices"
]

def build_visit_index(chexpert, split):
    df = chexpert[["subject_id","study_id"] + LABELS]
    df = df.merge(split[["study_id","split"]], on="study_id", how="left")
    return df.drop_duplicates("study_id")
