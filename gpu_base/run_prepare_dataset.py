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
    print("Positive:", int(df["outcome"].sum()))

if __name__ == "__main__":
    main()
