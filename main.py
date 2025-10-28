import warnings, time
warnings.filterwarnings("ignore")

from src.config import Config
from src.utils import set_seed, get_device
from src.data import get_loaders
from src.model import make_model
from src.train import Trainer
import torch, wandb

from scipy.stats import pearsonr
import numpy as np

import pandas as pd, numpy as np, matplotlib.pyplot as plt
from src.plots import scatter, select_examples, plot_examples_from_df

if __name__ == "__main__":
    cfg = Config()
    set_seed(cfg.seed)
    device = get_device()

    t0 = time.time()

    train_loader, val_loader, test_loader = get_loaders(cfg)

    model = make_model().to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    trainer = Trainer(model, device, loss_fn, optimizer, project=cfg.project)
    trainer.train(train_loader, val_loader, epochs=cfg.epochs, best_path=cfg.best_path, last_path=cfg.last_path)

    model.load_state_dict(torch.load(cfg.best_path))
    test_metrics = trainer.test(test_loader)

    runtime = time.time() - t0
    print(f"runtime: {runtime/60:.2f} minutes")

    wandb.log({
        "loss/test": test_metrics["loss"],
        "pearson_r/test": test_metrics["pearson_r"],
        "runtime/minutes": runtime / 60,
    })
    
    ids, y_true_log, y_pred_log = trainer.predict(test_loader)
    pred_df = pd.DataFrame({
        "id": ids,
        "y_true_log": y_true_log,
        "y_pred_log": y_pred_log,
    })
    pred_df["y_true"] = np.power(10.0, pred_df["y_true_log"]) - 1
    pred_df["y_pred"] = np.power(10.0, pred_df["y_pred_log"]) - 1
    pred_df["abs_diff_log"] = (pred_df["y_true_log"] - pred_df["y_pred_log"]).abs()
    pred_df["abs_diff"] = (pred_df["y_true"] - pred_df["y_pred"]).abs()

    r_log, _ = pearsonr(y_true_log, y_pred_log)
    r_lin, _ = pearsonr(np.power(10, y_true_log) - 1, np.power(10, y_pred_log) - 1)
    print(f"log-space r: {r_log:.3f}, linear-space r: {r_lin:.3f}")

    wandb.log({"predictions_table": wandb.Table(dataframe=pred_df)})

    chart = scatter(y_true_log, y_pred_log)
    wandb.log({"scatter_test": wandb.Html(chart.to_html())})

    test_df = test_loader.dataset.df
    id_to_path = dict(zip(test_df["id"], test_df["h5path"]))
    best_df, worst_df = select_examples(pred_df, n=3)

    good_fig = plot_examples_from_df(best_df, id_to_path, "Best Predictions")
    bad_fig = plot_examples_from_df(worst_df, id_to_path, "Worst Predictions")

    wandb.log({
        "examples/best": wandb.Image(good_fig),
        "examples/worst": wandb.Image(bad_fig),
    })

    plt.close(good_fig)
    plt.close(bad_fig)
    wandb.finish()
