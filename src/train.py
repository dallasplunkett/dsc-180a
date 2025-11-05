import torch, numpy as np
from scipy.stats import pearsonr
import wandb
from tqdm import tqdm

def pearson_r(y_log, p_log):
    return pearsonr(y_log, p_log)[0] if len(y_log) > 1 else np.nan

class Trainer:
    def __init__(self, model, device, loss_fn, optimizer, project="cnn"):
        self.model = model
        self.device = device
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.best_val_loss = float("inf")

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
        return {"loss": total_loss / n, "pearson_r": pearson_r(y, p)}

    def train(self, train_loader, val_loader, epochs: int,
              best_path="best_model.pth", last_path="last_model.pth"):
        for epoch in tqdm(range(1, epochs + 1), desc="epochs", leave=True):
            tr = self._run(train_loader, train=True,  desc="train")
            va = self._run(val_loader,   train=False, desc="validation")

            wandb.log({
                "epoch": epoch,
                "loss/train": tr["loss"],
                "pearson_r/train": tr["pearson_r"],
                "loss/validation": va["loss"],
                "pearson_r/validation": va["pearson_r"],
            })
            if va["loss"] < self.best_val_loss:
                self.best_val_loss = va["loss"]
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
