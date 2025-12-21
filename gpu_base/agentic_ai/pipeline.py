import torch
import timm
from PIL import Image
import torchvision.transforms as T
from agentic_ai.llm.compute_sc import compute_sc_batch
from agentic_ai.dcs.compute_rf import compute_rf
from agentic_ai.utils import load_cfg
from agentic_ai.vision.model import DualViewSwin
import pandas as pd

cfg = load_cfg("./agentic_ai/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model(
    cfg["model"]["swin_model_name"],
    pretrained=False,
    num_classes=0
)

out_dim = len(cfg["data"]["label_columns"])
model = DualViewSwin(backbone, out_dim=out_dim).to(device)
model.load_state_dict(torch.load(cfg["paths"]["checkpoint"], map_location=device))
model.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)

def extract_cm_batch(frontal_paths, lateral_paths):
    frontal_batch = torch.cat([load_image(p) for p in frontal_paths], dim=0)
    lateral_batch = torch.cat([load_image(p) for p in lateral_paths], dim=0)
    with torch.no_grad():
        x_f = model.backbone(frontal_batch)
        x_l = model.backbone(lateral_batch)
        cm_batch = (x_f + x_l) / 2
    return cm_batch


def run_pipeline_batch(frontal_paths, lateral_paths, llm_text_batch, dcs_weights, device="cuda"):
    cm_batch = extract_cm_batch(frontal_paths, lateral_paths)
    sc_dicts = compute_sc_batch(llm_text_batch)
    sc_batch = [float(d["score"]) for d in sc_dicts]
    rf_batch = compute_rf(cm_batch, sc_dicts, dcs_weights, device=device)
    results = []
    for i in range(len(cm_batch)):
        results.append({
            "Cm_vector": cm_batch[i].cpu().tolist(),
            "Sc": sc_batch[i],
            "Rf": rf_batch[i].item()
        })
    df = pd.DataFrame(results)
    df.to_csv("./results/pipeline_output.csv", index=False)
    return results
