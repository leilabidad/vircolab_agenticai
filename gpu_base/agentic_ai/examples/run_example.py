import torch
from agentic_ai.pipeline import run_pipeline
import json

frontal_img = torch.randn(1, 3, 224, 224)
lateral_img = torch.randn(1, 3, 224, 224)
llm_text = "Patient has shortness of breath and mild fever."

dcs_weights = {
    "w1": 0.4,
    "w2": 0.3,
    "w3": 0.2,
    "intercept": 0.1
}

result = run_pipeline(frontal_img, lateral_img, llm_text, dcs_weights)
print(json.dumps(result, indent=4))
