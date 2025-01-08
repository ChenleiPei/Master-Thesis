import torch
import numpy as np
import random


def set_random_seed(seed=None):

    if seed is None:
        torch.random.seed()
    else:
        torch.manual_seed(seed)

    np.random.seed(seed)
    random.seed(seed)