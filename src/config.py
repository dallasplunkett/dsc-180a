from pathlib import Path

remote_cfg = {
    "project": "remote",
    "model": "resnet18",
    # Training
    "size": 1024,
    "batch_size": 4,
    "epochs": 16,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    # System
    "num_workers": 1,
    "pin_memory": True,
    # Data
    "image_dir": str(Path.home() / "teams" / "b1"),
    "train_csv": str(Path.home() / "teams" / "b1" / "BNPP_DT_train_with_ages.csv"),
    "val_csv": str(Path.home() / "teams" / "b1" / "BNPP_DT_val_with_ages.csv"),
    "test_csv": str(Path.home() / "teams" / "b1" / "BNPP_DT_test_with_ages.csv"),
}

local_cfg = {
    "project": "local",
    "model": "resnet18",
    # Training
    "size": 128,
    "batch_size": 64,
    "epochs": 10,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    # System
    "num_workers": 0,
    "pin_memory": False,
    # Data
    "image_dir": "data",
    "train_csv": "data/BNPP_DT_train_with_ages.csv",
    "val_csv": "data/BNPP_DT_val_with_ages.csv",
    "test_csv": "data/BNPP_DT_test_with_ages.csv",
}

test_cfg = {
    "project": "test",
    "model": "resnet18",
    # Training
    "size": 64,
    "batch_size": 128,
    "epochs": 4,
    "learning_rate": 1e-5,
    "weight_decay": 1e-5,
    # System
    "num_workers": 0,
    "pin_memory": False,
    # Data
    "image_dir": "data",
    "train_csv": "data/BNPP_DT_train_with_ages.csv",
    "val_csv": "data/BNPP_DT_val_with_ages.csv",
    "test_csv": "data/BNPP_DT_test_with_ages.csv",
}
