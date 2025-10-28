from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Project
    project: str = "cnn"
    seed: int | None = None

    # Data paths
    data_dir: Path = Path("data")
    train_csv: Path = Path("data/train.csv")
    test_csv: Path = Path("data/test.csv")

    # Model + training
    size: int = 512
    val_frac: float = 0.30
    batch_size: int = 16
    epochs: int = 50
    lr: float = 1e-5
    weight_decay: float = 1e-5

    # System
    num_workers: int = 0
    pin_memory: bool = False

    # Checkpoints
    best_path: Path = Path("checkpoints/best_model.pth")
    last_path: Path = Path("checkpoints/last_model.pth")
