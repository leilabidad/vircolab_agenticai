import torch
from agentic_ai.vision.model import DualViewSwin
from agentic_ai.vision.utils import load_cfg
from agentic_ai.llm.compute_sc import compute_sc
from agentic_ai.dcs.compute_rf import compute_rf
import timm

cfg = load_cfg("./agentic_ai/config.yaml")
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

backbone = timm.create_model("swin_base_patch4_window7_224", pretrained=False, num_classes=0)
model = DualViewSwin(backbone, out_dim=len(cfg["data"]["label_columns"])).to(device)
model.load_state_dict(torch.load(cfg["paths"]["checkpoint"], map_location=device))
model.eval()

def extract_cm(frontal_img, lateral_img):
    frontal = frontal_img.to(device)
    lateral = lateral_img.to(device)
    with torch.no_grad():
        x_f = model.backbone(frontal)
        x_l = model.backbone(lateral)
        cm = (x_f + x_l) / 2
    return cm.squeeze().cpu().tolist()

def run_pipeline(frontal_img, lateral_img, llm_text, dcs_weights):
    cm_vector = extract_cm(frontal_img, lateral_img)
    sc_json = compute_sc(llm_text)
    sc_score = sc_json.get("score", 0.0)
    rf = compute_rf(cm_vector, sc_score, dcs_weights)
    return {
        "Cm_vector": cm_vector,
        "Sc": sc_json,
        "Rf": rf
    }
