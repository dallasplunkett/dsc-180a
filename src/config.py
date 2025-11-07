from dataclasses import dataclass
from pathlib import Path

@dataclass
class Config:
    # --- Project ---
    project: str     = "remote"
    seed: int | None = None

    # --- Data Paths ---
    data_dir: Path  = Path.home() / "teams" / "b1"
    train_csv: Path = data_dir / "BNPP_DT_train_with_ages.csv"
    val_csv: Path   = data_dir / "BNPP_DT_val_with_ages.csv"
    test_csv: Path  = data_dir / "BNPP_DT_test_with_ages.csv"

    # --- Model & Training ---
    size: int          = 1024
    batch_size: int    = 16
    epochs: int        = 20
    lr: float           = 1e-5
    weight_decay: float = 1e-5

    # --- System ---
    num_workers: int = 4
    pin_memory: bool = True

    # --- Checkpoints ---
    best_path: Path = Path("checkpoints/best_model.pth")
    last_path: Path = Path("checkpoints/last_model.pth")

@dataclass
class TestConfig:
    # --- Project ---
    project: str     = "local"
    seed: int | None = None

    # --- Data Paths ---
    data_dir: Path  = Path("data")
    train_csv: Path = data_dir / "BNPP_DT_train_with_ages.csv"
    val_csv: Path   = data_dir / "BNPP_DT_val_with_ages.csv"
    test_csv: Path  = data_dir / "BNPP_DT_test_with_ages.csv"

    # --- Model & Training ---
    size: int          = 256
    batch_size: int    = 32
    epochs: int        = 8
    lr: float           = 1e-5
    weight_decay: float = 1e-5

    # --- System ---
    num_workers: int = 0
    pin_memory: bool = False

    # --- Checkpoints ---
    best_path: Path = Path("checkpoints/best_model.pth")
    last_path: Path = Path("checkpoints/last_model.pth")
