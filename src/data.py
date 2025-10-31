import h5py, torch, pandas as pd
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
import torch.nn.functional as F

def resize(cfg, x):
    return F.interpolate(
        x.unsqueeze(0),
        size=(cfg.size, cfg.size),
        mode="bilinear",
        align_corners=False
    ).squeeze(0)

def normalize(x):
    return (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-6)

class Image(Dataset):
    def __init__(self, cfg, df):
        self.df = df.reset_index(drop=True)
        self.cfg = cfg
        self.file_cache = {}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        if row.h5path not in self.file_cache:
            self.file_cache[row.h5path] = h5py.File(row.h5path, "r")
        file = self.file_cache[row.h5path]
        img = file[row.id][()]
        x = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        x = resize(self.cfg, x)
        x = normalize(x)
        y = torch.tensor(row.bnpp_log, dtype=torch.float32)

        return x, y, row.id

def get_df(cfg, csv_attr="train_csv"):
    csv_path = getattr(cfg, csv_attr)
    df = pd.read_csv(csv_path, usecols=["unique_key", "bnpp_value_log"])
    df.columns = ["id", "bnpp_log"]
    df = df.astype({"id": str, "bnpp_log": "float32"})

    id_set = set(df["id"])
    rows = []
    for path in cfg.data_dir.glob("*.hdf5"):
        try:
            with h5py.File(path, "r") as file:
                rows.extend(
                    {"id": k, "h5path": str(path)}
                    for k in file.keys()
                    if k in id_set
                )
        except OSError as e:
            print(f"failed read: {path} | error: {e}")

    return df.merge(pd.DataFrame(rows), on="id", how="inner")

def get_loaders(cfg):
    full_df = get_df(cfg, "train_csv")
    test_df = get_df(cfg, "test_csv")
    train_df, val_df = train_test_split(
        full_df,
        test_size=cfg.val_frac,
        random_state=cfg.seed
    )
    n_train, n_val, n_test = len(train_df), len(val_df), len(test_df)
    n = n_train + n_val + n_test
    print(
        f"n: {n:,} | "
        f"train: {n_train:,} ({n_train/n:.1%}) | "
        f"val: {n_val:,} ({n_val/n:.1%}) | "
        f"test: {n_test:,} ({n_test/n:.1%})"
    )
    opts = dict(
        batch_size=cfg.batch_size,
        num_workers=cfg.num_workers,
        pin_memory=cfg.pin_memory
    )
    loaders = {
        "train": DataLoader(Image(cfg, train_df), shuffle=True,  **opts),
        "val": DataLoader(Image(cfg, val_df), shuffle=False, **opts),
        "test": DataLoader(Image(cfg, test_df), shuffle=False, **opts),
    }
    return loaders["train"], loaders["val"], loaders["test"]
