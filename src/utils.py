import random, torch, numpy as np

def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"seed: {seed}")

def get_device(verbose=True):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    if verbose: print(f"device: {device}")
    return torch.device(device)

def log_config(cfg, device=None):
    print("=" * 60)
    print("RUN CONFIGURATION")
    print("=" * 60)
    if device:
        print(f"{'device':20s}: {device}")
    for k, v in vars(cfg).items():
        print(f"{k:20s}: {v}")
    print("=" * 60 + "\n")
