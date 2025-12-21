import yaml

def load_cfg(path):
    with open(path) as f:
        return yaml.safe_load(f)
