import torch.nn as nn


class DualViewSwin(nn.Module):
    def __init__(self, backbone, out_dim):
        super().__init__()
        self.backbone = backbone
        self.head = nn.Linear(backbone.num_features, out_dim)

    def forward(self, x_fr, x_la):
        e1 = self.backbone(x_fr)
        e2 = self.backbone(x_la)
        emb = (e1 + e2) / 2
        out = self.head(emb)
        return out, emb
