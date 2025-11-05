import warnings, time
warnings.filterwarnings("ignore")

from src.config import Config, TestConfig
from src.utils import set_seed, get_device, get_pred_df, scatter, examples
from src.data import get_loaders
from src.model import make_model
from src.train import Trainer
import torch, wandb
import matplotlib.pyplot as plt

if __name__ == "__main__":
    # --- Setup --- 
    cfg = TestConfig()
    cfg.seed = set_seed(cfg.seed)
    device = get_device()
    wandb.init(
        project=cfg.project,
        config=vars(cfg),
        name=f"{cfg.project}_{cfg.seed}"
    )

    # --- Data ---
    train_loader, val_loader, test_loader = get_loaders(cfg)

    # --- Model ---
    model = make_model().to(device)
    loss_fn = torch.nn.L1Loss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
    )
    trainer = Trainer(model, device, loss_fn, optimizer, project=cfg.project)

    # --- Fit & Eval ---
    t0 = time.time()
    trainer.train(
        train_loader, val_loader,
        epochs=cfg.epochs,
        best_path=cfg.best_path, last_path=cfg.last_path
    )
    model.load_state_dict(torch.load(cfg.best_path))
    test_metrics = trainer.test(test_loader)
    runtime = time.time() - t0
    wandb.log({
        "loss/test": test_metrics["loss"],
        "pearson_r/test": test_metrics["pearson_r"],
        "runtime/minutes": runtime / 60,
    })
    
    # --- Predict ---
    ids, y_true_log, y_pred_log = trainer.predict(test_loader)
    pred_df = get_pred_df(ids, y_true_log, y_pred_log)
    wandb.log({
        "predictions": wandb.Table(dataframe=pred_df)
    })
    chart = scatter(y_true_log, y_pred_log)
    wandb.log({
        "scatter": wandb.Html(chart.to_html())
    })
    good_example, bad_example = examples(pred_df, test_loader.dataset.df)
    wandb.log({
        "examples/best": wandb.Image(good_example),
        "examples/worst": wandb.Image(bad_example),
    })

    # --- Cleanup ---
    plt.close(good_example)
    plt.close(bad_example)
    wandb.finish()
