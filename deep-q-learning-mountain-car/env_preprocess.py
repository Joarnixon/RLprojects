import numpy as np
import torch

def process_observation(obs):
    return torch.tensor(obs, dtype=torch.float32).unsqueeze(0)