from pathlib import Path

import numpy as np
import torch
from scipy.stats import pearsonr
from tqdm import tqdm

import wandb


class Trainer:
    def __init__(self, model, device, loss_fn, optimizer):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.best_valid_loss = float("inf")

    def _run(self, loader, train: bool, desc: str):
        self.model.train() if train else self.model.eval()
        total_loss, n = 0.0, 0
        ys, ps = [], []
        for xb, yb, _ in loader:
            xb, yb = xb.to(self.device), yb.to(self.device)
            with torch.set_grad_enabled(train):
                pred = self.model(xb).reshape(-1)
                loss = self.loss_fn(pred, yb)
                if train:
                    self.optimizer.zero_grad(set_to_none=True)
                    loss.backward()
                    self.optimizer.step()
            total_loss += loss.item() * xb.size(0)
            n += xb.size(0)
            ys.append(yb.detach().cpu().numpy())
            ps.append(pred.detach().cpu().numpy())
        y, p = np.concatenate(ys), np.concatenate(ps)
        return {
            "loss": total_loss / n,
            "pearson_r": pearsonr(y, p)[0] if len(y) > 1 else np.nan,
        }

    def train(
        self,
        train_loader,
        valid_loader,
        epochs: int,
        best_path="cnn/checkpoints/best_model.pth",
        last_path="cnn/checkpoints/last_model.pth",
    ):
        best_path = Path(best_path)
        last_path = Path(last_path)
        best_path.parent.mkdir(parents=True, exist_ok=True)
        last_path.parent.mkdir(parents=True, exist_ok=True)

        for _ in tqdm(range(1, epochs + 1), desc="epochs", leave=True):
            tr = self._run(train_loader, train=True, desc="train")
            va = self._run(valid_loader, train=False, desc="validation")

            wandb.log(
                {
                    "loss/train": tr["loss"],
                    "pearson_r/train": tr["pearson_r"],
                    "loss/validation": va["loss"],
                    "pearson_r/validation": va["pearson_r"],
                }
            )

            if va["loss"] < self.best_valid_loss:
                self.best_valid_loss = va["loss"]
                torch.save(self.model.state_dict(), best_path)

        torch.save(self.model.state_dict(), last_path)

    @torch.no_grad()
    def evaluate(self, loader, phase="validation"):
        return self._run(loader, train=False, desc=phase)

    @torch.no_grad()
    def test(self, loader):
        return self.evaluate(loader, phase="test")

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        ids, ys, ps = [], [], []
        for xb, yb, idb in loader:
            xb = xb.to(self.device)
            pred = self.model(xb).reshape(-1)
            ps.append(pred.detach().cpu().numpy())
            ys.append(yb.cpu().numpy())
            ids.extend(idb)
        return np.array(ids), np.concatenate(ys), np.concatenate(ps)
