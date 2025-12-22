import yaml
from src.loader import load_tables
from src.image_selector import select_images
from src.builder import build_dataset

def main():
    with open("config/config.yaml") as f:
        cfg = yaml.safe_load(f)
    chexpert, metadata, split = load_tables(cfg)
    frontal, lateral = select_images(metadata, cfg["paths"]["images"])
    df = build_dataset(chexpert, split, frontal, lateral, cfg)
    df.to_pandas().to_csv(cfg["paths"]["output"], index=False)
    print("DONE")
    print("Visits:", len(df))
    print("Frontal:", df["path_img_fr"].notna().sum())
    print("Lateral:", df["path_img_la"].notna().sum())
    print("Positive:", int(df["outcome"].sum()))

if __name__ == "__main__":
    main()