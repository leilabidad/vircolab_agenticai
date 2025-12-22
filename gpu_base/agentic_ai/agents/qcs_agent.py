import torch

class QCSAgent:
    def __init__(self, threshold):
        self.threshold = threshold

    def run(self, rf, issues):
        qc_flag = rf >= self.threshold
        if qc_flag:
            return {
                "qc_flag": True,
                "action": "manual_review",
                "issues": issues
            }
        return {
            "qc_flag": False,
            "action": "accept",
            "issues": issues
        }
