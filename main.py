import warnings
warnings.filterwarnings("ignore")
import os
import random
import numpy as np
import pandas as pd
import h5py
from tqdm import tqdm
import torch
from torch import nn
from torch.utils.data import TensorDataset, DataLoader
import torchvision.models as tvm
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr
import wandb
import altair as alt

alt.data_transformers.disable_max_rows()

config = {
    "data_dir": os.getenv("DATA_DIR", "/app/data"),
    "size": 256,
    "val_frac": 0.10,
    "test_frac": 0.10,
    "batch_size": 32,
    "epochs": 12,
    "lr": 2e-4,
    "weight_decay": 1e-4,
    "workers": min(8, os.cpu_count() or 2),
}

def seed_everything(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"Using seed: {seed}")
    return seed

def get_device():
    return torch.device("cpu")

def load_data(dir):
    dir = os.path.expanduser(dir)
    meas_dfs = []
    for p in tqdm(os.listdir(dir), desc="Loading CSV files"):
        if p.endswith(".csv"):
            df = pd.read_csv(os.path.join(dir, p)).rename(columns={"unique_key": "id", "bnpp_value_log": "bnpp_log"})[["id", "bnpp_log"]]
            meas_dfs.append(df)
    if not meas_dfs:
        raise ValueError("No CSV files loaded")
    
    meas_df = pd.concat(meas_dfs, ignore_index=True)
    img_rows = []
    for p in tqdm(os.listdir(dir), desc="Loading HDF5 files"):
        if p.endswith(".hdf5"):
            with h5py.File(os.path.join(dir, p), "r") as f:
                for k in f.keys():
                    img_rows.append({"id": k, "image": f[k][()]})
    if not img_rows:
        raise ValueError("No HDF5 files loaded")
    
    img_df = pd.DataFrame(img_rows)
    return img_df.merge(meas_df, on="id", how="inner")

def preprocess_images(images, size):
    images = torch.from_numpy(images)[:, None]
    images = torch.nn.functional.interpolate(images, size=(size, size), mode="bilinear")
    max_vals = images.max(dim=-1, keepdim=True)[0].max(dim=-2, keepdim=True)[0]
    max_vals = torch.clamp(max_vals, min=1e-6)
    return images / max_vals

def prepare_tensors(df, size):
    images = np.stack(df["image"].values)
    X = preprocess_images(images, size).numpy()
    y = df["bnpp_log"].values.astype(np.float32)
    return torch.from_numpy(X), torch.from_numpy(y)

def make_loaders(df, config, seed):
    test_size = config["test_frac"]
    val_size = config["val_frac"] / (1.0 - test_size)

    trainval_df, test_df = train_test_split(df, test_size=test_size, shuffle=True, random_state=seed)
    train_df, val_df = train_test_split(trainval_df, test_size=val_size, shuffle=True, random_state=seed)

    Xtr, ytr = prepare_tensors(train_df, config["size"])
    Xva, yva = prepare_tensors(val_df, config["size"])
    Xte, yte = prepare_tensors(test_df, config["size"])

    loader_opts = {
        "batch_size": config["batch_size"],
        "num_workers": config["workers"],
    }
    train_loader = DataLoader(TensorDataset(Xtr, ytr), shuffle=True, **loader_opts)
    val_loader = DataLoader(TensorDataset(Xva, yva), shuffle=False, **loader_opts)
    test_loader = DataLoader(TensorDataset(Xte, yte), shuffle=False, **loader_opts)
    
    return train_loader, val_loader, test_loader

def base_resnet18(in_ch=1):
    model = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    w = model.conv1.weight.data.mean(dim=1, keepdim=True)
    model.conv1 = nn.Conv2d(in_ch, 64, kernel_size=7, stride=2, padding=3, bias=False)
    model.conv1.weight.data = w
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model

class ResNetTrainer:
    def __init__(self, model, loss_fn, optimizer, config):
        self.model = model
        self.config = config
        self.device = get_device()
        self.model.to(self.device)
        self.loss_fn = loss_fn
        self.optimizer = optimizer

    def train_one_epoch(self, loader):
        self.model.train()
        total_loss = 0.0
        ys_log = []
        ps_log = []
        for xb, yb in tqdm(loader, desc="Training"):
            xb, yb = xb.to(self.device), yb.to(self.device).reshape(-1)
            self.optimizer.zero_grad()
            pred = self.model(xb).reshape(-1)
            loss = self.loss_fn(pred, yb)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
            ys_log.append(yb.cpu().numpy())
            ps_log.append(pred.detach().cpu().numpy())
        y_log = np.concatenate(ys_log)
        p_log = np.concatenate(ps_log)
        mae_log = np.mean(np.abs(y_log - p_log))
        y_bnpp = np.exp(y_log) - 1
        p_bnpp = np.exp(p_log) - 1
        pearson_r, _ = pearsonr(y_bnpp, p_bnpp) if len(y_bnpp) > 1 else (np.nan, np.nan)
        return total_loss / len(loader.dataset), mae_log, pearson_r

    def evaluate(self, loader, phase="val"):
        self.model.eval()
        ys_log = []
        ps_log = []
        with torch.no_grad():
            for xb, yb in loader:
                xb = xb.to(self.device)
                pred = self.model(xb).reshape(-1).detach().cpu().numpy()
                ys_log.append(yb.reshape(-1).numpy())
                ps_log.append(pred)
        
        y_log = np.concatenate(ys_log)
        p_log = np.concatenate(ps_log)
        
        mae_log = np.mean(np.abs(y_log - p_log))
        
        y_bnpp = np.exp(y_log) - 1
        p_bnpp = np.exp(p_log) - 1
        
        r, _ = pearsonr(y_bnpp, p_bnpp) if len(y_bnpp) > 1 else (np.nan, np.nan)
        
        metrics = {
            "mae_log": mae_log,
            "pearson_r": r,
        }
        
        return metrics, (y_log, p_log)

def create_scatter_plot(y_log_true, y_log_pred):
    y = np.clip(np.exp(y_log_true) - 1, 1e-6, None)
    p = np.clip(np.exp(y_log_pred) - 1, 1e-6, None)
    lo = 1.0
    hi = float(10**np.ceil(np.log10(max(y.max(), p.max(), 10))))
    lx, ly = np.log10(y), np.log10(p)
    r, _ = pearsonr(lx, ly) if len(y) > 1 else (np.nan, None)
    slope, itc = np.polyfit(lx, ly, 1)
    xx = np.array([lo, hi])
    yy = 10**(slope * np.log10(xx) + itc)

    dfp = pd.DataFrame({"Measured": y, "Predicted": p})
    dfi = pd.DataFrame({"Measured": xx, "Predicted": xx})
    dff = pd.DataFrame({"Measured": xx, "Predicted": yy})

    base = (
        alt.Chart(dfp)
        .mark_point(size=35, opacity=0.8)
        .encode(
            x=alt.X("Measured:Q", scale=alt.Scale(type="log", domain=[lo, hi]), title="Measured BNPP (pg/mL)"),
            y=alt.Y("Predicted:Q", scale=alt.Scale(type="log", domain=[lo, hi]), title="Inferred BNPP (pg/mL)"),
        )
        .properties(width=420, height=420, title=f"r = {r:.3f}")
    )
    identity = (
        alt.Chart(dfi)
        .mark_line(strokeDash=[4, 4], color="black")
        .encode(x="Measured:Q", y="Predicted:Q")
    )
    fit = (
        alt.Chart(dff)
        .mark_line(color="red")
        .encode(x="Measured:Q", y="Predicted:Q")
    )
    return base + identity + fit

if __name__ == "__main__":
    seed = seed_everything()

    df = load_data(config["data_dir"])
    train_loader, val_loader, test_loader = make_loaders(df, config, seed)

    base_model = base_resnet18()
    model = ResNetTrainer(
        base_model,
        nn.L1Loss(),
        torch.optim.AdamW(
            base_model.parameters(),
            lr=config["lr"],
            weight_decay=config["weight_decay"]
        ),
        config,
    )

    wandb.init(project="dsc-180a", config=config)

    for epoch in range(1, config["epochs"] + 1):
        tr_loss, tr_mae_log, tr_pearson_r = model.train_one_epoch(train_loader)
        val_metrics, (val_y_log, val_p_log) = model.evaluate(val_loader, phase="val")
        wandb.log({
            "epoch": epoch,
            "loss/train": tr_loss,
            "mae_log/train": tr_mae_log,
            "pearson_r/train": tr_pearson_r,
            "loss/val": val_metrics["mae_log"],
            "mae_log/val": val_metrics["mae_log"],
            "pearson_r/val": val_metrics["pearson_r"],
        })

    test_metrics, (test_y_log, test_p_log) = model.evaluate(test_loader, phase="test")
    wandb.log({
        "mae_log/test": test_metrics["mae_log"],
        "pearson_r/test": test_metrics["pearson_r"],
    })

    chart = create_scatter_plot(test_y_log, test_p_log)
    wandb.log({"scatter_test": wandb.Html(chart.to_html())})
    wandb.finish()
