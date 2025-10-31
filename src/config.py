from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Project
    project: str = "all_data_locally"
    seed: int | None = None

    # Data Paths
    data_dir: Path = Path.home() / "Downloads" / "data"
    train_csv: Path = data_dir / "train.csv"
    test_csv: Path = data_dir / "test.csv"

    # Model & Training
    size: int = 1024
    val_frac: float = 0.30
    batch_size: int = 4
    epochs: int = 30
    lr: float = 1e-5
    weight_decay: float = 1e-5

    # System
    num_workers: int = 0
    pin_memory: bool = False

    # Checkpoints
    best_path: Path = Path("checkpoints/best_model.pth")
    last_path: Path = Path("checkpoints/last_model.pth")
