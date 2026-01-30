from pathlib import Path

import h5py
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def resize(config, x):
    return F.interpolate(
        x.unsqueeze(0),
        size=(config["size"], config["size"]),
        mode="bilinear",
        align_corners=False,
    ).squeeze(0)


def normalize(x):
    return (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-6)


class Image(Dataset):
    def __init__(self, config, df):
        self.df = df.reset_index(drop=True)
        self.config = config

    def __len__(self):
        return len(self.df)

    def __getitem__(self, i):
        row = self.df.iloc[i]
        with h5py.File(row.h5path, "r") as file:
            img = file[row.id][()]  # pyright: ignore[reportIndexIssue]
        x = torch.as_tensor(img, dtype=torch.float32).unsqueeze(0)
        x = resize(self.config, x)
        x = normalize(x)
        y = torch.tensor(row.bnpp_log, dtype=torch.float32)

        return x, y, row.id


def get_df(config, csv_attr="train_csv"):
    csv_path = Path(config[csv_attr])
    df = pd.read_csv(csv_path, usecols=["id", "bnpp_log"])
    df = df.astype({"id": str, "bnpp_log": "float32"})

    id_set = set(df["id"])
    rows = []
    image_dir = Path(config["image_dir"])
    for path in image_dir.glob("*.hdf5"):
        try:
            with h5py.File(path, "r") as file:
                rows.extend(
                    {"id": k, "h5path": str(path)} for k in file.keys() if k in id_set
                )
        except OSError as e:
            print(f"failed read: {path} | error: {e}")

    return df.merge(pd.DataFrame(rows), on="id", how="inner")


def get_loaders(config):
    train_df = get_df(config, "train_csv")
    valid_df = get_df(config, "valid_csv")
    test_df = get_df(config, "test_csv")

    opts = dict(
        batch_size=config["batch_size"],
        num_workers=config["num_workers"],
        pin_memory=config["pin_memory"],
    )

    loaders = {
        "train": DataLoader(Image(config, train_df), shuffle=True, **opts),
        "valid": DataLoader(Image(config, valid_df), shuffle=False, **opts),
        "test": DataLoader(Image(config, test_df), shuffle=False, **opts),
    }

    return loaders["train"], loaders["valid"], loaders["test"]
