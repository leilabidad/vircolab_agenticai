import torch
import timm
import json
from PIL import Image
import torchvision.transforms as T
from pathlib import Path

from agentic_ai.utils import load_cfg
from agentic_ai.llm.compute_sc import compute_sc_batch
from agentic_ai.dcs.compute_rf import compute_rf
from agentic_ai.agents.qcs_agent import QCSAgent
from agentic_ai.agents.decision_manager import DecisionManager

cfg = load_cfg("./agentic_ai/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model(
    cfg["model"]["swin_model_name"],
    pretrained=False,
    num_classes=0
).to(device)

state = torch.load(cfg["paths"]["checkpoint"], map_location="cpu")
backbone.load_state_dict({k.replace("backbone.", ""): v for k, v in state.items() if k.startswith("backbone.")})
backbone.eval()

transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor()
])

qcs_agent = QCSAgent(threshold=0.7)
decision_manager = DecisionManager()

def load_image(path):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0).to(device)


def extract_cm_batch(frontal_paths, lateral_paths):
    cm_list = []
    for fr_path, la_path in zip(frontal_paths, lateral_paths):
        fr_img = load_image(fr_path)
        la_img = load_image(la_path)

        imgs = torch.cat([fr_img, la_img], dim=0)

        with torch.no_grad():
            feats = backbone(imgs)
            cm = feats.mean(dim=0, keepdim=True)

        cm_list.append(cm.detach())

        del fr_img, la_img, imgs, feats, cm
        torch.cuda.empty_cache()

    return torch.cat(cm_list, dim=0)


def run_pipeline_batch(frontal_paths, lateral_paths, llm_text_batch, dcs_weights):
    cm_batch = extract_cm_batch(frontal_paths, lateral_paths)
    sc_outputs = compute_sc_batch(llm_text_batch)
    rf_batch = compute_rf(cm_batch, sc_outputs, dcs_weights)
    results = []

    for i in range(len(rf_batch)):
        rf = rf_batch[i].item() if torch.is_tensor(rf_batch[i]) else float(rf_batch[i])
        sc = float(sc_outputs[i]["score"])
        qc = qcs_agent.run(rf, sc_outputs[i].get("issues", []))
        label = decision_manager.decide(rf)
        results.append({
            "patient_index": i,
            "pred_label": label,
            "Cm": cm_batch[i].mean().item(),
            "Sc": sc,
            "Rf": rf,
            "QC_flag": qc["qc_flag"],
            "issues": qc["issues"]
        })

    Path("./results").mkdir(parents=True, exist_ok=True)
    with open("./results/final_output.json", "w") as f:
        json.dump(results, f, indent=2)

    return results
