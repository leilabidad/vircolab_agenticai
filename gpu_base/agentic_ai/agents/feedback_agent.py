class FeedbackAgent:
    def __init__(self):
        self.history = []

    def record(self, cm, sc, rf, uncertainty):
        self.history.append({
            "cm": cm.detach().cpu().numpy(),
            "sc": sc,
            "rf": rf,
            "uncertainty": uncertainty
        })


    def decide(self, qc_flag, uncertainty):
        if qc_flag and uncertainty > 0.2:
            return "re-evaluate"
        return "accept"
    
    def update(self, cm, sc, rf, uncertainty):
        self.history.append({
            "cm": cm.detach().cpu(),
            "sc": sc,
            "rf": rf,
            "uncertainty": uncertainty
        })
