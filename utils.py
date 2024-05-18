import os
import torch


def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)

def save_model(model, path):
    print(f"Saving model to {path}")
    torch.save(model.state_dict(), path)

def load_model(model, path):
    print(f"Loading model from {path}")
    model.load_state_dict(torch.load(path))
    return model



