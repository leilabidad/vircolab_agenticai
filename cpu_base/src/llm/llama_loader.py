import torch
from transformers import AutoTokenizer, AutoModel

class LlamaEncoder:
    def __init__(self, model_name="meta-llama/Llama-2-7b-hf", device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.model.eval()

    def encode_text(self, text):
        inputs = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=512).to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)
        last_hidden = outputs.hidden_states[-1].squeeze(0)
        vector = last_hidden.mean(dim=0)
        return vector
