import random, torch
import numpy as np

def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    return seed

def get_device():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    return torch.device(device)

def log_config(cfg, device=None):
    print("RUN CONFIGURATION")
    print("=" * 60)
    if device:
        print(f"{'device':20s}: {device}")
    for k, v in vars(cfg).items():
        print(f"{k:20s}: {v}")
    print("=" * 60, "\n")

def log_data_summary(train_loader, val_loader, test_loader):
    print("DATA SUMMARY")
    print("=" * 60)

    n_train = len(train_loader.dataset)
    n_val   = len(val_loader.dataset)
    n_test  = len(test_loader.dataset)
    n_total = n_train + n_val + n_test

    print(f"{'total samples':20s}: {n_total:,}")
    print(f"{'train samples':20s}: {n_train:,} ({n_train/n_total:.1%})")
    print(f"{'validation samples':20s}: {n_val:,} ({n_val/n_total:.1%})")
    print(f"{'test samples':20s}: {n_test:,} ({n_test/n_total:.1%})")

    xb, _, _ = next(iter(train_loader))
    print(f"{'input shape':20s}: {tuple(xb.shape)}")

    print("=" * 60, "\n")
