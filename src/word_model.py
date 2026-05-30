"""CNN models for word-level sign video classification."""

from __future__ import annotations

import torch
from torch import nn


class TemporalBlock(nn.Module):
    """Residual temporal convolution block over per-frame features."""

    def __init__(self, channels: int, dilation: int, dropout_rate: float) -> None:
        super().__init__()
        padding = dilation
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size=3, padding=padding, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


class WordSignNet(nn.Module):
    """Frame CNN with temporal CNN pooling for larger word vocabularies."""

    architecture = "temporal_cnn_v2"

    def __init__(self, num_classes: int, dropout_rate: float = 0.4) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("WordSignNet requires at least one class")
        if not 0.3 <= dropout_rate <= 0.5:
            raise ValueError("dropout_rate must be in [0.3, 0.5]")

        self.frame_features = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(128, 192, kernel_size=3, padding=1),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.frame_projection = nn.Sequential(
            nn.Flatten(),
            nn.Linear(192, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.temporal = nn.Sequential(
            TemporalBlock(256, dilation=1, dropout_rate=dropout_rate),
            TemporalBlock(256, dilation=2, dropout_rate=dropout_rate),
            TemporalBlock(256, dilation=4, dropout_rate=dropout_rate),
        )
        self.attention = nn.Sequential(
            nn.Conv1d(256, 128, kernel_size=1),
            nn.Tanh(),
            nn.Conv1d(128, 1, kernel_size=1),
        )
        self.classifier = nn.Sequential(
            nn.Linear(512, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass.

        Args:
            x: Tensor with shape ``(batch, frames, 1, image_size, image_size)``.
        """
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError(f"Expected input shape (B, T, 1, H, W), got {tuple(x.shape)}")
        batch, frames = x.shape[:2]
        encoded = self.frame_features(x.reshape(batch * frames, *x.shape[2:]))
        encoded = self.frame_projection(encoded).reshape(batch, frames, 256)
        temporal = self.temporal(encoded.transpose(1, 2))
        attention_weights = torch.softmax(self.attention(temporal), dim=2)
        attended = (temporal * attention_weights).sum(dim=2)
        pooled = temporal.max(dim=2).values
        return self.classifier(torch.cat((attended, pooled), dim=1))

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        """Predict the most likely word class for one prepared video tensor."""
        class_index, confidence, _ = self.predict_with_margin(x)
        return class_index, confidence

    def predict_with_margin(self, x: torch.Tensor) -> tuple[int, float, float]:
        """Predict the top class plus its confidence gap over the runner-up."""
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            top_values, top_indices = probs.topk(k=min(2, probs.shape[1]), dim=1)
        confidence = float(top_values[0, 0].item())
        runner_up = float(top_values[0, 1].item()) if top_values.shape[1] > 1 else 0.0
        return int(top_indices[0, 0].item()), confidence, confidence - runner_up


class LegacyWordSignNet(nn.Module):
    """Previous compact word CNN kept so old checkpoints still load."""

    architecture = "summary_cnn_v1"

    def __init__(self, num_classes: int, dropout_rate: float = 0.4) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("LegacyWordSignNet requires at least one class")
        if not 0.3 <= dropout_rate <= 0.5:
            raise ValueError("dropout_rate must be in [0.3, 0.5]")

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=5, padding=2),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128, 192),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(192, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 5 or x.shape[2] != 1:
            raise ValueError(f"Expected input shape (B, T, 1, H, W), got {tuple(x.shape)}")
        frames = x.squeeze(2)
        appearance = frames.mean(dim=1)
        span = frames.max(dim=1).values - frames.min(dim=1).values
        if frames.shape[1] > 1:
            motion = frames[:, 1:].sub(frames[:, :-1]).abs().mean(dim=1)
        else:
            motion = torch.zeros_like(appearance)
        summary = torch.stack((appearance, motion, span), dim=1)
        return self.classifier(self.features(summary))

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        class_index, confidence, _ = self.predict_with_margin(x)
        return class_index, confidence

    def predict_with_margin(self, x: torch.Tensor) -> tuple[int, float, float]:
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            top_values, top_indices = probs.topk(k=min(2, probs.shape[1]), dim=1)
        confidence = float(top_values[0, 0].item())
        runner_up = float(top_values[0, 1].item()) if top_values.shape[1] > 1 else 0.0
        return int(top_indices[0, 0].item()), confidence, confidence - runner_up


WORD_MODEL_TYPES = (WordSignNet, LegacyWordSignNet)
