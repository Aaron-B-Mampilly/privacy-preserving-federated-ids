"""Global random-seed control for reproducibility.

The frozen spec requires every experiment to be reproducible from a
config + seed, and final results to be reported as mean +/- std over
3 seeds. This module is the single place that seeds every RNG the
project touches (Python's random, NumPy, and PyTorch/CUDA).
"""

import os
import random

import numpy as np
import torch


def set_seed(seed: int) -> None:
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    # Deterministic cuDNN kernels trade some speed for exact reproducibility,
    # which matters more here than raw training throughput.
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
