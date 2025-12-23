import json
from pathlib import Path

class TraceStore:
    def __init__(self, path):
        self.path = Path(path)
        self.path.mkdir(parents=True, exist_ok=True)

    def write(self, trace):
        with open(self.path / "trace.json", "a") as f:
            f.write(json.dumps(trace) + "\n")
