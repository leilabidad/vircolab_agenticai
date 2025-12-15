import torch

def save_checkpoint(data, path):
    torch.save(data, path)
