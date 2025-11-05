import h5py
import torch
import numpy as np
import pandas as pd
from scipy.stats import pearsonr

import altair as alt
import matplotlib.pyplot as plt
alt.data_transformers.disable_max_rows()

def scatter(y_log_true, y_log_pred):
    lx, ly = y_log_true, y_log_pred
    r, _ = pearsonr(lx, ly) if len(lx) > 1 else (np.nan, None)
    slope, itc = np.polyfit(lx, ly, 1)

    y = np.clip(np.power(10.0, y_log_true) - 1.0, 1e-6, None)
    p = np.clip(np.power(10.0, y_log_pred) - 1.0, 1e-6, None)
    lo, hi = 1.0, 100_000.0
    ticks = [10, 100, 1_000, 10_000, 100_000]

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
        .properties(width=420, height=420, title=f"r = {r:.3f}")
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