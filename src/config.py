from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # Project
    project: str = "remote"
    seed: int | None = None

    # Data Paths
    data_dir: Path = Path.home() / "teams" / "b1"
    train_csv: Path = data_dir / "BNPP_DT_train_with_ages.csv"
    val_csv: Path   = data_dir / "BNPP_DT_val_with_ages.csv"
    test_csv: Path  = data_dir / "BNPP_DT_test_with_ages.csv"

    # Model & Training
    size: int = 256
    batch_size: int = 16
    epochs: int = 5
    lr: float = 1e-5
    weight_decay: float = 1e-5

    # System
    num_workers: int = 4
    pin_memory: bool = True

    # Checkpoints
    best_path: Path = Path("checkpoints/best_model.pth")
    last_path: Path = Path("checkpoints/last_model.pth")
