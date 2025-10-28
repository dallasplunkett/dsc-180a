import h5py, torch, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

# --- Image helpers ---
def normalize(x):
    return (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-6)

def resize(x, size):
    return F.interpolate(x.unsqueeze(0), size=size, mode="bilinear", align_corners=False).squeeze(0)

def transform_fn(x, size):
    """Top-level transform (picklable)"""
    return normalize(resize(x, (size, size)))

# --- CSV + HDF5 scanning ---
def get_labels(csv_path, data_dir):
    df = pd.read_csv(csv_path, usecols=["unique_key", "bnpp_value_log"])
    df.columns = ["id", "bnpp_log"]
    df = df.astype({"id": str, "bnpp_log": "float32"})

    files = [f for f in Path(data_dir).glob("*.hdf5")]
    rows, id_set = [], set(df["id"])
    for path in files:
        with h5py.File(path, "r") as f:
            for key in f.keys():
                if key in id_set:
                    rows.append({"id": key, "h5path": str(path)})

    return df.merge(pd.DataFrame(rows), on="id", how="inner")

# --- Dataset with caching ---
class Image(Dataset):
    def __init__(self, df, size):
        self.df = df.reset_index(drop=True)
        self.size = size
        self.file_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if row.h5path not in self.file_cache:
            self.file_cache[row.h5path] = h5py.File(row.h5path, "r")
        f = self.file_cache[row.h5path]
        img = f[row.id][()]
        x = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        x = transform_fn(x, self.size)
        y = torch.tensor(row.bnpp_log, dtype=torch.float32)
        return x, y, row.id

# --- Loader creation ---
def get_loaders(cfg):
    full_df = get_labels(cfg.train_csv, cfg.data_dir)
    test_df = get_labels(cfg.test_csv, cfg.data_dir)
    train_df, val_df = train_test_split(full_df, test_size=cfg.val_frac, random_state=cfg.seed)

    opts = dict(batch_size=cfg.batch_size, num_workers=cfg.num_workers, pin_memory=cfg.pin_memory)
    loaders = {
        "train": DataLoader(Image(train_df, cfg.size), shuffle=True, **opts),
        "val":   DataLoader(Image(val_df, cfg.size),   shuffle=False, **opts),
        "test":  DataLoader(Image(test_df, cfg.size),  shuffle=False, **opts),
    }

    print({k: len(v.dataset) for k, v in loaders.items()})
    return loaders["train"], loaders["val"], loaders["test"]
