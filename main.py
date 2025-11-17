import time

import matplotlib.pyplot as plt
import torch

import wandb
from src.config import remote_cfg, test_cfg
from src.data import get_loaders
from src.model import make_model
from src.train import Trainer
from src.utils import examples, get_device, get_pred_df, parse_config, scatter

if __name__ == "__main__":
    # --- Setup ---
    cfg, project = parse_config(default_preset="remote")
    run = wandb.init(
        project=project,
        config=cfg,
        config_exclude_keys=["image_dir", "train_csv", "val_csv", "test_csv"],
    )
    device = get_device()

    # --- Data ---
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # --- Model ---
    model = make_model(name=cfg["model"]).to(device)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg["learning_rate"],
        weight_decay=cfg["weight_decay"],
    )
    trainer = Trainer(model, device, loss_fn, optimizer)

    # --- Train ---
    t0 = time.time()
    trainer.train(train_loader, val_loader, epochs=cfg["epochs"])

    # --- Test ---
    model.load_state_dict(torch.load("checkpoints/best_model.pth"))
    test_metrics = trainer.test(test_loader)
    wandb.log(
        {
            "loss/test": test_metrics["loss"],
            "pearson_r/test": test_metrics["pearson_r"],
            "runtime/minutes": (time.time() - t0) / 60,
        }
    )

    # --- Prediction Table ---
    ids, yt, yp = trainer.predict(test_loader)
    pred_df = get_pred_df(ids, yt, yp)
    wandb.log({"predictions": wandb.Table(dataframe=pred_df)})

    # --- Predicted BNPP vs Actual BNPP, Pearson r ---
    scatter_plt = scatter(pred_df["y_true_log"].values, pred_df["y_pred_log"].values)
    wandb.log({"scatter": wandb.Html(scatter_plt.to_html())})

    # --- Best and Worst Predictions ---
    good, bad = examples(pred_df, test_loader.dataset.df)
    wandb.log({"examples/best": wandb.Image(good), "examples/worst": wandb.Image(bad)})

    plt.close(good)
    plt.close(bad)
    wandb.finish()
