"""Training pipeline for the SignNet ASL classifier."""

from __future__ import annotations

import csv
import sys
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset

from src.model import SignNet, index_to_letter, label_to_index
from src.preprocessor import Preprocessor


def _ensure_optimizer_dynamo_compatibility() -> None:
    """Install a tiny Dynamo shim when this PyTorch build has a broken import."""
    global torch
    try:
        import torch._dynamo  # noqa: F401
    except Exception:
        dynamo = types.ModuleType("torch._dynamo")

        def disable(fn=None, recursive=True):
            if fn is None:
                return lambda wrapped: wrapped
            return fn

        def graph_break():
            return None

        dynamo.disable = disable
        dynamo.graph_break = graph_break
        sys.modules["torch._dynamo"] = dynamo
        torch._dynamo = dynamo


class SignMNISTDataset(Dataset):
    """PyTorch dataset for Sign Language MNIST CSV files."""

    def __init__(self, csv_path: str | Path, augment: bool = False, augmentation_config: dict | None = None) -> None:
        """Load samples from a Sign Language MNIST CSV file.

        Args:
            csv_path: Path to a CSV with ``label`` plus 784 pixel columns.
            augment: Whether to apply training augmentation.
            augmentation_config: Augmentation settings from config.
        """
        self.csv_path = Path(csv_path)
        data = np.loadtxt(self.csv_path, delimiter=",", skiprows=1, dtype=np.uint16)
        labels = data[:, 0].astype(int)
        pixels = data[:, 1:].astype(np.uint8).reshape(-1, 28, 28)

        keep_indices: list[int] = []
        compact_labels: list[int] = []
        for idx, label in enumerate(labels):
            try:
                compact_labels.append(label_to_index(int(label)))
                keep_indices.append(idx)
            except ValueError:
                continue

        self.images = pixels[keep_indices]
        self.labels = np.asarray(compact_labels, dtype=np.int64)
        self.augment = augment
        self.augmentation_config = augmentation_config or {}

    def __len__(self) -> int:
        """Return the number of usable static-letter samples."""
        return int(len(self.labels))

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Return one normalized image tensor and compact class label."""
        image = Preprocessor.prepare_dataset_image(
            self.images[index],
            augment=self.augment,
            augmentation_config=self.augmentation_config,
        )
        return image, torch.tensor(int(self.labels[index]), dtype=torch.long)


class Trainer:
    """Train, evaluate, log, and checkpoint SignNet models."""

    def __init__(self, config: dict[str, Any]) -> None:
        """Create a trainer from validated config."""
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        model_cfg = config["model"]
        self.model = SignNet(
            num_classes=int(model_cfg["num_classes"]),
            dropout_rate=float(model_cfg["dropout_rate"]),
        ).to(self.device)

    def train(self) -> dict[str, Any]:
        """Run the configured training loop.

        Returns:
            Dictionary containing final accuracy and checkpoint metadata.
        """
        train_cfg = self.config["training"]
        train_dataset = SignMNISTDataset(
            self.config["dataset"]["train_csv"],
            augment=True,
            augmentation_config=self.config["augmentation"],
        )
        test_dataset = SignMNISTDataset(self.config["dataset"]["test_csv"], augment=False)
        train_loader = DataLoader(
            train_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            test_dataset,
            batch_size=int(train_cfg["batch_size"]),
            shuffle=False,
            num_workers=0,
        )

        criterion = nn.CrossEntropyLoss()
        _ensure_optimizer_dynamo_compatibility()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=max(float(train_cfg["weight_decay"]), 1e-4),
        )
        scheduler = self._build_scheduler(optimizer)
        self._prepare_training_log()

        final_metrics: dict[str, Any] = {"overall_accuracy": 0.0, "per_class_accuracy": {}}
        for epoch in range(1, int(train_cfg["epochs"]) + 1):
            train_loss = self.train_epoch(train_loader, criterion, optimizer)
            try:
                final_metrics = self.evaluate(test_loader)
                val_accuracy = float(final_metrics["overall_accuracy"])
            except Exception as exc:
                print(f"Evaluation was not completed: {exc}")
                val_accuracy = float(final_metrics.get("overall_accuracy", 0.0))

            if isinstance(scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                scheduler.step(val_accuracy)
            else:
                scheduler.step()
            self._append_training_log(epoch, train_loss, val_accuracy)
            self.save_if_better(val_accuracy)
            print(f"Epoch [{epoch}/{train_cfg['epochs']}], Loss: {train_loss:.4f}, Val Accuracy: {val_accuracy:.2f}%")

        self._print_accuracy_metrics(final_metrics)
        final_metrics["checkpoint_saved"] = Path(self.config["model"]["checkpoint_path"]).exists()
        return final_metrics

    def train_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        """Train the model for one epoch and return mean loss."""
        self.model.train()
        running_loss = 0.0
        for images, labels in loader:
            images = images.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(images), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        return running_loss / max(1, len(loader))

    def evaluate(self, loader: DataLoader) -> dict[str, Any]:
        """Evaluate the model and return overall plus per-class accuracy."""
        self.model.eval()
        correct = 0
        total = 0
        class_correct = [0] * 24
        class_total = [0] * 24
        with torch.no_grad():
            for images, labels in loader:
                images = images.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(images).argmax(dim=1)
                matches = predictions.eq(labels)
                total += int(labels.numel())
                correct += int(matches.sum().item())
                for label, match in zip(labels.cpu().tolist(), matches.cpu().tolist()):
                    class_total[label] += 1
                    class_correct[label] += int(match)

        overall = 100.0 * correct / total if total else 0.0
        per_class = {
            index_to_letter(i): (100.0 * class_correct[i] / class_total[i] if class_total[i] else 0.0)
            for i in range(24)
        }
        return {"overall_accuracy": overall, "per_class_accuracy": per_class}

    def save_if_better(self, new_accuracy: float) -> bool:
        """Save the checkpoint when ``new_accuracy`` is at least the previous best."""
        checkpoint_path = Path(self.config["model"]["checkpoint_path"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        previous_accuracy = 0.0
        if checkpoint_path.exists():
            try:
                checkpoint = torch.load(checkpoint_path, map_location=self.device)
                previous_accuracy = float(checkpoint.get("accuracy", 0.0)) if isinstance(checkpoint, dict) else 0.0
            except Exception:
                previous_accuracy = 0.0

        if float(new_accuracy) >= previous_accuracy:
            torch.save(
                {
                    "model_state_dict": self.model.state_dict(),
                    "accuracy": float(new_accuracy),
                    "letters": "ABCDEFGHIKLMNOPQRSTUVWXY",
                },
                checkpoint_path,
            )
            print(f"Saved checkpoint to {checkpoint_path} ({new_accuracy:.2f}%)")
            return True
        print(f"Checkpoint not saved: {new_accuracy:.2f}% < previous {previous_accuracy:.2f}%")
        return False

    def _build_scheduler(self, optimizer: optim.Optimizer) -> optim.lr_scheduler.LRScheduler:
        train_cfg = self.config["training"]
        if str(train_cfg["scheduler"]) == "CosineAnnealingLR":
            return optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, int(train_cfg["epochs"])))
        return optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            patience=int(train_cfg["scheduler_patience"]),
            factor=float(train_cfg["scheduler_factor"]),
        )

    def _prepare_training_log(self) -> None:
        log_cfg = self.config["logging"]
        log_dir = Path(log_cfg["log_dir"])
        log_dir.mkdir(parents=True, exist_ok=True)
        path = log_dir / str(log_cfg["training_log_file"])
        with path.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(["epoch", "train_loss", "val_accuracy"])

    def _append_training_log(self, epoch: int, train_loss: float, val_accuracy: float) -> None:
        path = Path(self.config["logging"]["log_dir"]) / str(self.config["logging"]["training_log_file"])
        with path.open("a", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow([epoch, f"{train_loss:.6f}", f"{val_accuracy:.6f}"])

    def _print_accuracy_metrics(self, metrics: dict[str, Any]) -> None:
        print(f"Overall accuracy: {float(metrics.get('overall_accuracy', 0.0)):.2f}%")
        per_class = metrics.get("per_class_accuracy", {})
        if not per_class:
            print("Per-class accuracy: evaluation was not completed")
            return
        for letter, accuracy in per_class.items():
            print(f"{letter}: {float(accuracy):.2f}%")
