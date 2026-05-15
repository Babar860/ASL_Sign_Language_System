from pathlib import Path

import torch

from src.config import load_config
from src.trainer import SignMNISTDataset, Trainer


def test_dataset_loading_sample():
    dataset = SignMNISTDataset("archive/sign_mnist_train/sign_mnist_train.csv")
    image, label = dataset[0]
    assert len(dataset) > 0
    assert tuple(image.shape) == (1, 28, 28)
    assert 0 <= int(label.item()) <= 23


def test_save_if_better_saves_on_equal_accuracy(tmp_path: Path):
    config = load_config("config.yaml")
    config["model"]["checkpoint_path"] = str(tmp_path / "models" / "signnet_best.pth")
    trainer = Trainer(config)
    assert trainer.save_if_better(0.0) is True
    checkpoint = Path(config["model"]["checkpoint_path"])
    assert checkpoint.exists()
    first_mtime = checkpoint.stat().st_mtime
    assert trainer.save_if_better(0.0) is True
    assert checkpoint.stat().st_mtime >= first_mtime
    assert "model_state_dict" in torch.load(checkpoint, map_location="cpu")
