class AdaptiveQCAgent:
    def __init__(self, base_threshold=0.7):
        self.base_threshold = base_threshold

    def evaluate(self, rf, uncertainty):
        threshold = self.base_threshold + uncertainty
        qc_flag = rf < threshold
        return qc_flag, threshold
