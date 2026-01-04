import yaml
from src.builder import build_dataset

def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)

    df = build_dataset(cfg)
    df.to_csv(cfg["paths"]["output"], index=False)

    print("DONE")
    print("Visits:", len(df))
    print("Frontal:", df["path_img_fr"].notna().sum())
    print("Lateral:", df["path_img_la"].notna().sum())
    print("Patients:", df["subject_id"].nunique())

    # Outcome stats
    num_positive = int(df["outcome"].sum(skipna=True))  # جمع فقط ردیف های غیر خالی
    num_zero = (df["outcome"] == 0).sum()
    num_missing = df["outcome"].isna().sum()

    print("Positive (outcome=1):", num_positive)
    print("Zero (outcome=0):", num_zero)
    print("Missing (outcome=NaN):", num_missing)




if __name__ == "__main__":
    main()
