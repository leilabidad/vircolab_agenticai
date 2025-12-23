import json
from pathlib import Path
from datetime import datetime

class TraceStore:
    def __init__(self, root="./traces"):
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    def log(self, record):
        ts = datetime.utcnow().isoformat()
        path = self.root / f"{ts}.json"
        with open(path, "w") as f:
            json.dump(record, f, indent=2)
