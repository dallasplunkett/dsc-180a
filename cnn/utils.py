import argparse

import altair as alt
import h5py
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from scipy.stats import pearsonr

from cnn.config import local_config, remote_config, test_config

alt.data_transformers.disable_max_rows()


def get_device():
    device = (
        "cuda"
        if torch.cuda.is_available()
        else "mps"
        if torch.backends.mps.is_available()
        else "cpu"
    )
    return torch.device(device)


def get_pred_df(id, bnpp_log, predicted_bnpp_log):
    pred_df = pd.DataFrame(
        {
            "id": id,
            "bnpp_log": bnpp_log,
            "predicted_bnpp_log": predicted_bnpp_log,
        }
    )
    pred_df["bnpp"] = np.power(10.0, pred_df["bnpp_log"]) - 1.0
    pred_df["predicted_bnpp"] = np.power(10.0, pred_df["predicted_bnpp_log"]) - 1.0
    pred_df["abs_diff_log"] = (
        pred_df["bnpp_log"] - pred_df["predicted_bnpp_log"]
    ).abs()
    pred_df["abs_diff"] = (pred_df["bnpp"] - pred_df["predicted_bnpp"]).abs()
    return pred_df


def scatter(bnpp_log, predicted_bnpp_log):
    df = pd.DataFrame(
        {
            "bnpp_log": np.asarray(list(bnpp_log), dtype=float),
            "predicted_bnpp_log": np.asarray(list(predicted_bnpp_log), dtype=float),
        }
    )

    df = df[np.isfinite(df["bnpp_log"]) & np.isfinite(df["predicted_bnpp_log"])].copy()

    r = pearsonr(df["bnpp_log"], df["predicted_bnpp_log"])[0] if len(df) > 1 else np.nan

    lo = float(min(df["bnpp_log"].min(), df["predicted_bnpp_log"].min()))  # pyright: ignore[reportArgumentType]
    hi = float(max(df["bnpp_log"].max(), df["predicted_bnpp_log"].max()))  # pyright: ignore[reportArgumentType]

    x = alt.X("bnpp_log:Q", title="True BNPP (log10)", scale=alt.Scale(domain=[lo, hi]))
    y = alt.Y(
        "predicted_bnpp_log:Q",
        title="Predicted BNPP (log10)",
        scale=alt.Scale(domain=[lo, hi]),
    )

    base = (
        alt.Chart(df)
        .mark_point(size=35, opacity=0.4)
        .encode(x=x, y=y)
        .properties(width=420, height=420, title=f"r = {r:.3f}")
    )

    identity = (
        alt.Chart(pd.DataFrame({"bnpp_log": [lo, hi], "predicted_bnpp_log": [lo, hi]}))
        .mark_line(color="black", strokeDash=[4, 4])
        .encode(x=x, y=y)
    )

    fit = (
        alt.Chart(df)
        .transform_regression("bnpp_log", "predicted_bnpp_log", method="linear")
        .mark_line(color="red")
        .encode(x=x, y=y)
    )

    return base + identity + fit


@torch.no_grad()
def get_images_by_ids(id_to_path, ids):
    imgs = []
    for id in ids:
        with h5py.File(id_to_path[id], "r") as f:
            imgs.append(f[id][()])  # pyright: ignore[reportIndexIssue]
    return imgs


def plot_examples_from_df(df, rows, title):
    imgs = get_images_by_ids(rows, df["id"])
    bnpp = df["bnpp"].to_numpy()
    predicted_bnpp = df["predicted_bnpp"].to_numpy()
    ids = df["id"].to_numpy()

    n = len(df)
    fig, axes = plt.subplots(1, n, figsize=(4 * n, 4))
    for i, ax in enumerate(axes):
        ax.imshow(imgs[i], cmap="gray")
        ax.axis("off")
        ax.set_title(
            f"ID:{ids[i]}\nBNPP:{bnpp[i]:.1f}\nPredicted BNPP:{predicted_bnpp[i]:.1f}"
        )
    fig.suptitle(title)
    plt.tight_layout()
    return fig


def examples(pred_df, test_df, n=3):
    id_to_path = dict(zip(test_df["id"], test_df["h5path"]))
    best_df = pred_df.nsmallest(n, "abs_diff")
    worst_df = pred_df.nlargest(n, "abs_diff")

    good_fig = plot_examples_from_df(best_df, id_to_path, "Best Predictions")
    bad_fig = plot_examples_from_df(worst_df, id_to_path, "Worst Predictions")

    return good_fig, bad_fig


PRESETS = {
    "remote": remote_config,
    "local": local_config,
    "test": test_config,
}


def parse_config(default_preset: str = "test"):
    p = argparse.ArgumentParser()

    # Which base config to start from
    p.add_argument(
        "--preset",
        "--config",
        dest="preset",
        choices=list(PRESETS.keys()),
        help="Base configuration to use (remote/local/test)",
    )

    # High-level / experiment-level knobs
    p.add_argument("--project")
    p.add_argument("--model")

    # Core training knobs
    p.add_argument("--size", type=int)
    p.add_argument("--batch_size", type=int)
    p.add_argument("--epochs", type=int)
    p.add_argument("--learning_rate", type=float)
    p.add_argument("--weight_decay", type=float)

    args = p.parse_args()

    # 1) Choose base preset
    preset = args.preset or default_preset
    config = PRESETS[preset].copy()

    # 2) Project handling
    project = config.get("project", preset)
    if args.project is not None:
        project = args.project
        config["project"] = args.project

    # 3) Override other keys if specified
    for k, v in vars(args).items():
        if k in ("preset", "project"):
            continue
        if v is not None:
            config[k] = v

    return config, project
