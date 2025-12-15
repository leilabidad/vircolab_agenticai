import torch
from transformers import AutoTokenizer, AutoModel

class ClinicalNoteEncoder:
    def __init__(self, model_name, device, max_len):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name).to(device)
        self.max_len = max_len
        self.model.eval()

    def encode(self, texts):
        tok = self.tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=self.max_len,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model(**tok).last_hidden_state[:,0,:]

        return out.cpu()
