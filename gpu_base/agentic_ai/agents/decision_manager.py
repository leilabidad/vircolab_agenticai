class DecisionManager:
    def __init__(self):
        pass

    def decide(self, rf):
        if rf >= 0.8:
            return "High Risk"
        if rf >= 0.5:
            return "Medium Risk"
        return "Low Risk"
