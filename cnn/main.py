import time

import matplotlib.pyplot as plt
import torch

import wandb
from cnn.data import get_loaders
from cnn.model import make_model
from cnn.train import Trainer
from cnn.utils import examples, get_device, get_pred_df, parse_config, scatter

if __name__ == "__main__":
    # --- Setup ---
    config, project = parse_config(default_preset="remote")
    wandb_config = wandb.helper.parse_config(  # type: ignore
        config, exclude=("image_dir", "train_csv", "valid_csv", "test_csv")
    )
    run = wandb.init(
        project=project,
        config=wandb_config,
    )
    device = get_device()

    # --- Data ---
    train_loader, valid_loader, test_loader = get_loaders(config)

    # --- Model ---
    model = make_model(name=config["model"]).to(device)

    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config["learning_rate"],
        weight_decay=config["weight_decay"],
    )
    trainer = Trainer(model, device, loss_fn, optimizer)

    # --- Train ---
    t0 = time.time()
    trainer.train(train_loader, valid_loader, epochs=config["epochs"])

    # --- Test ---
    model.load_state_dict(
        torch.load("cnn/checkpoints/best_model.pth", map_location=device)
    )
    test_metrics = trainer.test(test_loader)
    wandb.log(
        {
            "loss/test": test_metrics["loss"],
            "pearson_r/test": test_metrics["pearson_r"],
            "runtime/minutes": (time.time() - t0) / 60,
        }
    )

    # --- Prediction Table ---
    ids, bnpp_log, predicted_bnpp_log = trainer.predict(test_loader)
    predictions = get_pred_df(ids, bnpp_log, predicted_bnpp_log)
    wandb.log({"predictions": wandb.Table(dataframe=predictions)})

    # --- Predicted BNPP vs Actual BNPP, Pearson r ---
    scatter_plt = scatter(
        predictions["bnpp_log"].values, predictions["predicted_bnpp_log"].values
    )
    wandb.log({"scatter": wandb.Html(scatter_plt.to_html())})

    # --- Best and Worst Predictions ---
    good, bad = examples(predictions, test_loader.dataset.df)  # type: ignore
    wandb.log({"examples/best": wandb.Image(good), "examples/worst": wandb.Image(bad)})

    plt.close(good)
    plt.close(bad)
    wandb.finish()
