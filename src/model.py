"""SignNet convolutional model for ASL static-letter classification."""

from __future__ import annotations

import torch
from torch import nn


STATIC_LETTERS = "ABCDEFGHIKLMNOPQRSTUVWXY"


class SignNet(nn.Module):
    """CNN classifier for 24 static ASL letter classes.

    Args:
        num_classes: Number of output classes. The application uses 24.
        dropout_rate: Dropout probability for the classifier layers.
    """

    def __init__(self, num_classes: int = 24, dropout_rate: float = 0.4) -> None:
        super().__init__()
        if num_classes != 24:
            raise ValueError("SignNet requires exactly 24 classes")
        if not 0.3 <= dropout_rate <= 0.5:
            raise ValueError("dropout_rate must be in [0.3, 0.5]")

        self.conv_block1 = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.conv_block3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 3 * 3, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(512, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Run a forward pass and return raw logits.

        Args:
            x: Tensor with shape ``(batch_size, 1, 28, 28)``.

        Returns:
            Logits with shape ``(batch_size, 24)``.

        Raises:
            ValueError: If the input tensor shape is invalid.
        """
        if x.ndim != 4 or tuple(x.shape[1:]) != (1, 28, 28):
            raise ValueError(f"Expected input shape (B, 1, 28, 28), got {tuple(x.shape)}")
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.conv_block3(x)
        return self.classifier(x)

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        """Predict the most likely static ASL class for one image batch.

        Args:
            x: Tensor with shape ``(1, 1, 28, 28)``.

        Returns:
            A ``(class_index, confidence)`` pair.
        """
        self.eval()
        with torch.no_grad():
            logits = self.forward(x)
            probs = torch.softmax(logits, dim=1)
            confidence, class_index = probs.max(dim=1)
        return int(class_index.item()), float(confidence.item())


def index_to_letter(index: int) -> str:
    """Convert a compact model class index to an ASL letter."""
    return STATIC_LETTERS[index]


def label_to_index(label: int) -> int:
    """Convert a Sign Language MNIST label to a compact model class index."""
    if label == 9 or label == 25 or label < 0 or label > 25:
        raise ValueError(f"Label {label} is not a static 24-class ASL label")
    return label if label < 9 else label - 1


def index_to_label(index: int) -> int:
    """Convert a compact model class index back to the dataset label value."""
    if index < 0 or index >= len(STATIC_LETTERS):
        raise ValueError(f"Class index {index} is out of range")
    return index if index < 9 else index + 1
