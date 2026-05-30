"""Training pipeline for word-level sign videos."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, Dataset, random_split

from src.keypoint_model import KeypointWordNet, MediaPipeKeypointExtractor, keypoint_cache_path
from src.preprocessor import Preprocessor
from src.trainer import _ensure_optimizer_dynamo_compatibility
from src.word_model import WordSignNet
from src.word_vocab import display_word_label, normalize_word_label


class WordVideoDataset(Dataset):
    """Dataset backed by a CSV manifest of sign word videos."""

    def __init__(
        self,
        manifest_path: str | Path,
        frame_count: int = 16,
        image_size: int = 64,
        validate_videos: bool = True,
        cache_videos: bool = True,
        use_bbox: bool = True,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.frame_count = int(frame_count)
        self.image_size = int(image_size)
        self.use_bbox = bool(use_bbox)
        self.samples: list[dict[str, Any]] = []
        self.cached_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.labels: list[str] = []
        self.label_to_index: dict[str, int] = {}

        rejected: list[tuple[Path, str, str]] = []
        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label = normalize_word_label(row["label"])
                path = Path(row["path"])
                if not path.is_absolute():
                    path = self.manifest_path.parent.parent.parent / path
                if validate_videos:
                    ok, reason = self._is_readable_video(path)
                    if not ok:
                        rejected.append((path, display_word_label(label), reason))
                        continue
                if label not in self.label_to_index:
                    self.label_to_index[label] = len(self.labels)
                    self.labels.append(display_word_label(label))
                self.samples.append(
                    {
                        "path": path,
                        "label_index": self.label_to_index[label],
                        "split": row.get("split", "train"),
                        "frame_start": int(row.get("frame_start") or 1),
                        "frame_end": int(row.get("frame_end") or -1),
                        "bbox": json.loads(row["bbox"]) if row.get("bbox") and self.use_bbox else None,
                    }
                )

        if not self.samples:
            raise ValueError(f"No word videos found in manifest: {self.manifest_path}")
        if rejected:
            self._write_rejected_log(rejected)
            print(f"Skipped {len(rejected)} unreadable word video(s). See {self._rejected_log_path()}")
        if cache_videos:
            self._cache_videos()

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if self.cached_samples:
            return self.cached_samples[index]
        sample = self.samples[index]
        video = Preprocessor.prepare_word_video(
            sample["path"],
            self.frame_count,
            self.image_size,
            frame_start=sample["frame_start"],
            frame_end=sample["frame_end"],
            bbox=sample["bbox"],
        )
        label_index = int(sample["label_index"])
        return video, torch.tensor(label_index, dtype=torch.long)

    def _cache_videos(self) -> None:
        for index, sample in enumerate(self.samples, start=1):
            video = Preprocessor.prepare_word_video(
                sample["path"],
                self.frame_count,
                self.image_size,
                frame_start=sample["frame_start"],
                frame_end=sample["frame_end"],
                bbox=sample["bbox"],
            )
            self.cached_samples.append((video, torch.tensor(int(sample["label_index"]), dtype=torch.long)))
            if index % 50 == 0 or index == len(self.samples):
                print(f"Cached {index}/{len(self.samples)} word videos")

    def _is_readable_video(self, path: Path) -> tuple[bool, str]:
        if not path.exists():
            return False, "missing file"
        capture = cv2.VideoCapture(str(path))
        if not capture.isOpened():
            return False, "could not open"
        readable_frames = 0
        try:
            while readable_frames < self.frame_count:
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                readable_frames += 1
        finally:
            capture.release()
        if readable_frames <= 0:
            return False, "no readable frames"
        return True, ""

    def _rejected_log_path(self) -> Path:
        return self.manifest_path.with_name(f"{self.manifest_path.stem}_rejected.csv")

    def _write_rejected_log(self, rejected: list[tuple[Path, str, str]]) -> None:
        with self._rejected_log_path().open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(["path", "label", "reason"])
            for path, label, reason in rejected:
                writer.writerow([path.as_posix(), label, reason])


class KeypointWordDataset(Dataset):
    """Dataset backed by cached MediaPipe keypoint sequences."""

    def __init__(
        self,
        manifest_path: str | Path,
        frame_count: int,
        cache_dir: str | Path,
        use_bbox: bool = False,
    ) -> None:
        self.manifest_path = Path(manifest_path)
        self.frame_count = int(frame_count)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.samples: list[dict[str, Any]] = []
        self.cached_samples: list[tuple[torch.Tensor, torch.Tensor]] = []
        self.labels: list[str] = []
        self.label_to_index: dict[str, int] = {}

        with self.manifest_path.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                label = normalize_word_label(row["label"])
                path = Path(row["path"])
                if not path.is_absolute():
                    path = self.manifest_path.parent.parent.parent / path
                if label not in self.label_to_index:
                    self.label_to_index[label] = len(self.labels)
                    self.labels.append(display_word_label(label))
                bbox = json.loads(row["bbox"]) if use_bbox and row.get("bbox") else None
                self.samples.append(
                    {
                        "path": path,
                        "label_index": self.label_to_index[label],
                        "split": row.get("split", "train"),
                        "frame_start": int(row.get("frame_start") or 1),
                        "frame_end": int(row.get("frame_end") or -1),
                        "bbox": bbox,
                    }
                )

        if not self.samples:
            raise ValueError(f"No word videos found in manifest: {self.manifest_path}")
        self._cache_keypoints()

    def __len__(self) -> int:
        return len(self.cached_samples)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        return self.cached_samples[index]

    def _cache_keypoints(self) -> None:
        extractor = MediaPipeKeypointExtractor(frame_count=self.frame_count)
        try:
            for index, sample in enumerate(self.samples, start=1):
                cache_path = keypoint_cache_path(
                    self.cache_dir,
                    sample["path"],
                    sample["frame_start"],
                    sample["frame_end"],
                    sample["bbox"],
                )
                if cache_path.exists():
                    sequence = np.load(cache_path)
                else:
                    sequence = extractor.video_keypoints(
                        sample["path"],
                        frame_start=sample["frame_start"],
                        frame_end=sample["frame_end"],
                        bbox=sample["bbox"],
                    )
                    np.save(cache_path, sequence)
                self.cached_samples.append(
                    (
                        torch.from_numpy(sequence.astype(np.float32)),
                        torch.tensor(int(sample["label_index"]), dtype=torch.long),
                    )
                )
                if index % 25 == 0 or index == len(self.samples):
                    print(f"Cached {index}/{len(self.samples)} keypoint sequences")
        finally:
            extractor.close()


class WordTrainer:
    """Train and checkpoint a word-level sign video classifier."""

    def __init__(self, config: dict[str, Any]) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model_cfg = config["model"]
        self.model_type = str(model_cfg.get("word_model_type", "cnn")).lower()
        if self.model_type == "keypoint":
            self.dataset = KeypointWordDataset(
                config["avatar"]["sign_to_text_manifest"],
                frame_count=int(model_cfg["word_frame_count"]),
                cache_dir=config["training"]["word_keypoints_dir"],
                use_bbox=bool(config["training"].get("word_use_bbox", False)),
            )
            self.model = KeypointWordNet(
                num_classes=len(self.dataset.labels),
                dropout_rate=float(model_cfg["dropout_rate"]),
            ).to(self.device)
        else:
            self.dataset = WordVideoDataset(
                config["avatar"]["sign_to_text_manifest"],
                frame_count=int(model_cfg["word_frame_count"]),
                image_size=int(model_cfg["word_image_size"]),
                validate_videos=bool(config["training"].get("word_validate_videos", True)),
                cache_videos=bool(config["training"].get("word_cache_videos", True)),
                use_bbox=bool(config["training"].get("word_use_bbox", True)),
            )
            self.model = WordSignNet(
                num_classes=len(self.dataset.labels),
                dropout_rate=float(model_cfg["dropout_rate"]),
            ).to(self.device)

    def train(self) -> dict[str, Any]:
        train_cfg = self.config["training"]
        split_strategy = str(train_cfg.get("word_split_strategy", "wlasl"))
        train_indices = [idx for idx, sample in enumerate(self.dataset.samples) if sample["split"] == "train"]
        val_indices = [idx for idx, sample in enumerate(self.dataset.samples) if sample["split"] in {"val", "test"}]
        if split_strategy == "wlasl" and train_indices and val_indices:
            train_dataset = torch.utils.data.Subset(self.dataset, train_indices)
            val_dataset = torch.utils.data.Subset(self.dataset, val_indices)
        else:
            val_size = max(1, int(len(self.dataset) * 0.2)) if len(self.dataset) > 1 else 0
            train_size = len(self.dataset) - val_size
            train_dataset, val_dataset = random_split(
                self.dataset,
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42),
            )

        batch_size = int(train_cfg.get("word_batch_size", train_cfg["batch_size"]))
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

        criterion = nn.CrossEntropyLoss()
        _ensure_optimizer_dynamo_compatibility()
        optimizer = optim.Adam(
            self.model.parameters(),
            lr=float(train_cfg["learning_rate"]),
            weight_decay=max(float(train_cfg["weight_decay"]), 1e-4),
        )

        best_accuracy = 0.0
        epochs = int(train_cfg.get("word_epochs", train_cfg["epochs"]))
        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, criterion, optimizer)
            accuracy = self.evaluate(val_loader)
            if accuracy >= best_accuracy:
                best_accuracy = accuracy
                self.save_checkpoint(accuracy)
            print(f"Word epoch [{epoch}/{epochs}], Loss: {train_loss:.4f}, Val Accuracy: {accuracy:.2f}%")

        return {"overall_accuracy": best_accuracy, "classes": self.dataset.labels}

    def _train_epoch(self, loader: DataLoader, criterion: nn.Module, optimizer: optim.Optimizer) -> float:
        self.model.train()
        running_loss = 0.0
        for videos, labels in loader:
            videos = videos.to(self.device)
            labels = labels.to(self.device)
            optimizer.zero_grad()
            loss = criterion(self.model(videos), labels)
            loss.backward()
            optimizer.step()
            running_loss += float(loss.item())
        return running_loss / max(1, len(loader))

    def evaluate(self, loader: DataLoader) -> float:
        self.model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for videos, labels in loader:
                videos = videos.to(self.device)
                labels = labels.to(self.device)
                predictions = self.model(videos).argmax(dim=1)
                correct += int(predictions.eq(labels).sum().item())
                total += int(labels.numel())
        return 100.0 * correct / total if total else 0.0

    def save_checkpoint(self, accuracy: float) -> None:
        checkpoint_path = Path(self.config["model"]["word_checkpoint_path"])
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "model_state_dict": self.model.state_dict(),
                "accuracy": float(accuracy),
                "labels": self.dataset.labels,
                "frame_count": int(self.config["model"]["word_frame_count"]),
                "image_size": int(self.config["model"]["word_image_size"]),
                "architecture": getattr(self.model, "architecture", "unknown"),
                "model_type": self.model_type,
            },
            checkpoint_path,
        )
        print(f"Saved word checkpoint to {checkpoint_path} ({accuracy:.2f}%)")


def main() -> None:
    from src.config import load_config

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    metrics = WordTrainer(load_config(config_path)).train()
    print(f"Trained {len(metrics['classes'])} word classes")


if __name__ == "__main__":
    main()
