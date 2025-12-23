import torch

class UncertaintyAgent:
    def __init__(self, alpha=0.5):
        self.alpha = alpha

    def estimate(self, cm, sc):
        cm_var = cm.var(dim=1).mean().item()
        sc_var = abs(0.5 - sc)
        return self.alpha * cm_var + (1 - self.alpha) * sc_var
