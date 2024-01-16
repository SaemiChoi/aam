import torch
import numpy as np
import random


def set_seed(seed: int, device: str) -> torch.Generator:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(seed)

    return gen