import warnings
warnings.filterwarnings("ignore")

import os, random, h5py
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from scipy.stats import pearsonr

import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as tvm

from tqdm import tqdm
import wandb
import altair as alt
import matplotlib.pyplot as plt
alt.data_transformers.disable_max_rows()

def set_seed(seed=None):
    if seed is None:
        seed = random.randint(0, 2**32 - 1)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.backends.mps.is_available():
        torch.mps.manual_seed(seed)
    print(f"seed: {seed}")

def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def scan_data(data_dir):
    data_dir = Path(os.path.expanduser(data_dir))
    csvs = [p for p in data_dir.iterdir() if p.suffix == ".csv"]
    h5s = [p for p in data_dir.iterdir() if p.suffix == ".hdf5"]
    frames = []
    for p in csvs:
        df = pd.read_csv(p).rename(columns={"unique_key": "id", "bnpp_value_log": "bnpp_log"})[["id", "bnpp_log"]]
        df["id"] = df["id"].astype(str)
        df["bnpp_log"] = df["bnpp_log"].astype(np.float32)
        frames.append(df)
    labels_df = pd.concat(frames, ignore_index=True).drop_duplicates("id")
    label_ids = set(labels_df["id"])
    rows = []
    for p in h5s:
        with h5py.File(p, "r") as f:
            for k in f.keys():
                if k in label_ids:
                    rows.append({"h5path": str(p), "id": k})
    print(f"total labels: {len(labels_df)}, matched labels: {len(rows)}")
    return labels_df, rows

def minmax_normalize(x):
    return (x - x.min()) / torch.clamp(x.max() - x.min(), min=1e-6)

def make_transforms(size):
    return T.Compose([
        T.Resize((size, size)),
        T.Lambda(minmax_normalize)
    ])

class ImageStream(Dataset):
    def __init__(self, rows, labels_df, transform=None):
        self.rows = list(rows)
        self.labels = dict(zip(labels_df["id"], labels_df["bnpp_log"]))
        self.transform = transform

    def __len__(self): return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        path, key = r["h5path"], r["id"]
        with h5py.File(path, "r") as f:
            arr = f[key][()]
        x = torch.as_tensor(arr, dtype=torch.float32).unsqueeze(0)
        x = self.transform(x) if self.transform else x
        y = torch.tensor(self.labels[key], dtype=torch.float32)
        return x, y, key

def make_loaders(labels_df, rows, cfg):
    train_val, test = train_test_split(
        rows, test_size=cfg["test_frac"],
        random_state=cfg["seed"],
        shuffle=True)
    train, val = train_test_split(
        train_val,
        test_size=cfg["val_frac"] / (1 - cfg["test_frac"]),
        random_state=cfg["seed"],
        shuffle=True)
    options = dict(
        batch_size=cfg["batch_size"],
        num_workers=cfg["workers"],
        pin_memory=True,
        persistent_workers=(cfg["workers"] > 0))
    train_loader = DataLoader(
        ImageStream(train, labels_df, transform=make_transforms(cfg["size"])),
        shuffle=True,
        **options)
    val_loader = DataLoader(
        ImageStream(val, labels_df, transform=make_transforms(cfg["size"])),
        shuffle=False,
        **options)
    test_loader = DataLoader(
        ImageStream(test, labels_df, transform=make_transforms(cfg["size"])),
        shuffle=False,
        **options)
    return train_loader, val_loader, test_loader

def make_backbone():
    m = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1)
    w = m.conv1.weight.data.mean(dim=1, keepdim=True)
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    m.conv1.weight.data = w
    m.fc = nn.Identity()
    return m

def make_head():
    return nn.Sequential(
        nn.Linear(512, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, 1)
    )

def make_model():
    backbone = make_backbone()
    head = make_head()
    return nn.Sequential(backbone, head)

def pearson_r(y_log, p_log):
    y, p = np.power(10.0, y_log) - 1, np.power(10.0, p_log) - 1
    if len(y) <= 1:
        return np.nan
    return pearsonr(y, p)[0]

def make_scatter(y_log_true, y_log_pred, title="r"):
    y = np.clip(np.power(10.0, y_log_true) - 1.0, 1e-6, None)
    p = np.clip(np.power(10.0, y_log_pred) - 1.0, 1e-6, None)

    lo, hi = 1.0, 100_000.0
    ticks = [10, 100, 1_000, 10_000, 100_000]

    lx, ly = np.log10(y), np.log10(p)
    r, _ = pearsonr(lx, ly) if len(y) > 1 else (np.nan, None)
    slope, itc = np.polyfit(lx, ly, 1)
    xx = np.array([lo, hi])
    yy = 10 ** (slope * np.log10(xx) + itc)

    dfp = pd.DataFrame({"Measured BNPP": y, "Predicted BNPP": p})
    dfi = pd.DataFrame({"Measured BNPP": xx, "Predicted BNPP": xx})
    dff = pd.DataFrame({"Measured BNPP": xx, "Predicted BNPP": yy})

    xenc = alt.X(
        "Predicted BNPP:Q",
        scale=alt.Scale(type="log", domain=[lo, hi]),
        axis=alt.Axis(grid=False, values=ticks)
    )
    yenc = alt.Y(
        "Measured BNPP:Q",
        scale=alt.Scale(type="log", domain=[lo, hi]),
        axis=alt.Axis(grid=False, values=ticks)
    )

    base = (
        alt.Chart(dfp)
        .mark_point(size=35, opacity=0.8)
        .encode(x=xenc, y=yenc)
        .properties(width=420, height=420, title=f"{title} = {r:.3f}")
    )

    identity = alt.Chart(dfi).mark_line(strokeDash=[4,4], color="black").encode(x=xenc, y=yenc)
    fit = alt.Chart(dff).mark_line(color="red").encode(x=xenc, y=yenc)

    return base + identity + fit

def select_examples(pred_df, n=3):
    best = pred_df.nsmallest(n, "abs_diff").copy()
    worst = pred_df.nlargest(n, "abs_diff").copy()
    return best, worst


@torch.no_grad()
def get_images_by_ids(id_to_path, ids):
    imgs = []
    for id in ids:
        with h5py.File(id_to_path[id], "r") as f:
            imgs.append(f[id][()])
    return imgs

def plot_examples_from_df(df, rows, title):
    imgs = get_images_by_ids(rows, df["id"])
    y_true = df["y_true"].to_numpy()
    y_pred = df["y_pred"].to_numpy()
    ids = df["id"].to_numpy()

    n = len(df)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
        ax.set_title(f"id:{ids[i]}\nT:{y_true[i]:.1f}\nP:{y_pred[i]:.1f}")
    fig.suptitle(title)
    plt.tight_layout()
    return fig

def train_one_epoch(model, loader, device, optimizer, loss_fn):
    model.train()
    loss_sum, n_samples = 0.0, 0
    ys, ps = [], []
    for xb, yb, _ in tqdm(loader, desc="train"):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).reshape(-1)
        loss = loss_fn(pred, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        loss_sum += float(loss) * xb.size(0)
        n_samples += xb.size(0)
        ys.append(yb.detach().to("cpu").numpy())
        ps.append(pred.detach().to("cpu").numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return loss_sum / n_samples, pearson_r(y, p)

@torch.no_grad()
def evaluate(model, loader, device, loss_fn, phase="validation"):
    model.eval()
    loss_sum, n_samples = 0.0, 0
    ys, ps = [], []
    for xb, yb, _ in tqdm(loader, desc=phase):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).reshape(-1)
        loss = loss_fn(pred, yb)
        loss_sum += float(loss) * xb.size(0)
        n_samples += xb.size(0)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
    y, p = np.concatenate(ys), np.concatenate(ps)
    return {"loss": loss_sum / n_samples, "pearson_r": pearson_r(y, p)}

@torch.no_grad()
def collect_test_predictions(model, loader, device):
    model.eval()
    ids, ys, ps = [], [], []
    for xb, yb, idb in tqdm(loader, desc="test_predictions"):
        xb, yb = xb.to(device), yb.to(device)
        pred = model(xb).reshape(-1)
        ys.append(yb.cpu().numpy())
        ps.append(pred.cpu().numpy())
        ids.extend(idb)
    return ids, np.concatenate(ys), np.concatenate(ps)

if __name__ == "__main__":
    cfg = dict(
        data_dir=os.getenv("DATA_DIR", "data"),
        size=256,
        val_frac=0.10,
        test_frac=0.10,
        batch_size=16,
        epochs=50,
        lr=1e-5,
        weight_decay=1e-5,
        workers=min(8, os.cpu_count() or 2),
        project="cnn",
        seed=None,
        patience=3
    )
    set_seed(cfg["seed"])
    device = get_device()
    labels_df, images = scan_data(cfg["data_dir"])
    id_to_path = {r["id"]: r["h5path"] for r in images}

    train_loader, val_loader, test_loader = make_loaders(labels_df, images, cfg)
    model = make_model().to(device)
    loss_fn = nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["lr"],
        weight_decay=cfg["weight_decay"])
    
    wandb.init(project=cfg["project"], config=cfg)
    wandb.define_metric("epoch")
    wandb.define_metric("loss/*", step_metric="epoch")
    wandb.define_metric("pearson_r/*", step_metric="epoch")

    best_val_loss = float("inf")
    patience_counter = 0
    for epoch in range(1, cfg["epochs"] + 1):
        print(f"epoch: {epoch}")
        tr_loss, tr_r = train_one_epoch(model, train_loader, device, optimizer, loss_fn)
        val_metrics = evaluate(model, val_loader, device, loss_fn, phase="validation")
        wandb.log({
            "epoch": epoch,
            "loss/train": tr_loss,
            "loss/validation": val_metrics["loss"],
            "pearson_r/train": tr_r,
            "pearson_r/validation": val_metrics["pearson_r"],
        })
        if val_metrics["loss"] < best_val_loss:
            best_val_loss = val_metrics["loss"]
            patience_counter = 0
            torch.save(model.state_dict(), "best_model.pth")
        else:
            patience_counter += 1
            if patience_counter >= cfg["patience"]:
                print(f"early stop on epoch {epoch}")
                break

    model.load_state_dict(torch.load("best_model.pth"))
    test_metrics = evaluate(model, test_loader, device, loss_fn, phase="test")
    x_test, y_test_log, y_pred_log = collect_test_predictions(model, test_loader, device)
    wandb.log({
        "loss/test": test_metrics["loss"],
        "pearson_r/test": test_metrics["pearson_r"]
    })
    
    pred_df = pd.DataFrame({
        "id": x_test,
        "y_true_log": y_test_log,
        "y_pred_log": y_pred_log,
        "y_true": np.power(10.0, y_test_log) - 1,
        "y_pred": np.power(10.0, y_pred_log) - 1,
    })
    pred_df["abs_diff_log"] = (pred_df["y_true_log"] - pred_df["y_pred_log"]).abs()
    pred_df["abs_diff"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()
    wandb.log({"predictions_table": wandb.Table(dataframe=pred_df)})

    chart = make_scatter(y_test_log, y_pred_log, title="r")
    wandb.log({"scatter_test": wandb.Html(chart.to_html())})

    best_df, worst_df = select_examples(pred_df, n=3)
    good_fig = plot_examples_from_df(best_df, id_to_path, "Best Predictions")
    bad_fig = plot_examples_from_df(worst_df, id_to_path, "Worst Predictions")

    wandb.log({
        "examples/best": wandb.Image(good_fig),
        "examples/worst": wandb.Image(bad_fig)
    })

    plt.close("all")
    wandb.finish()
    if device.type in ["cuda", "mps"]:
        torch.cuda.empty_cache()
