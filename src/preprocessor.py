"""Image preprocessing utilities for training and live inference."""

from __future__ import annotations

import random

import cv2
import numpy as np
import torch


class Preprocessor:
    """Convert dataset images and webcam ROIs into SignNet tensors."""

    @staticmethod
    def prepare_dataset_image(
        image: np.ndarray,
        augment: bool = False,
        augmentation_config: dict | None = None,
    ) -> torch.Tensor:
        """Prepare a 28x28 dataset image for training or evaluation.

        Args:
            image: Grayscale image array.
            augment: Whether to apply random training augmentation.
            augmentation_config: Rotation, flip, brightness, and contrast settings.

        Returns:
            Float32 tensor with shape ``(1, 28, 28)`` and values in ``[0, 1]``.
        """
        if image is None:
            raise ValueError("image cannot be None")
        array = np.asarray(image)
        if array.ndim == 3:
            array = cv2.cvtColor(array, cv2.COLOR_BGR2GRAY)
        array = cv2.resize(array.astype(np.uint8), (28, 28), interpolation=cv2.INTER_AREA)

        if augment:
            array = Preprocessor._augment(array, augmentation_config or {})

        tensor = torch.from_numpy(array.astype(np.float32) / 255.0).unsqueeze(0)
        return tensor

    @staticmethod
    def prepare_frame(roi_array: np.ndarray) -> torch.Tensor:
        """Prepare a webcam ROI for inference.

        Args:
            roi_array: BGR, RGB, or grayscale image array of any non-empty size.

        Returns:
            Float32 tensor with shape ``(1, 1, 28, 28)``.
        """
        if roi_array is None or roi_array.size == 0:
            raise ValueError("roi_array must be a non-empty image")

        if roi_array.ndim == 3:
            gray = cv2.cvtColor(roi_array, cv2.COLOR_BGR2GRAY)
        elif roi_array.ndim == 2:
            gray = roi_array
        else:
            raise ValueError(f"Unsupported ROI shape: {roi_array.shape}")

        resized = cv2.resize(gray.astype(np.uint8), (28, 28), interpolation=cv2.INTER_AREA)
        return torch.from_numpy(resized.astype(np.float32) / 255.0).unsqueeze(0).unsqueeze(0)

    @staticmethod
    def _augment(image: np.ndarray, config: dict) -> np.ndarray:
        rotation = float(config.get("rotation_degrees", 0))
        flip = bool(config.get("horizontal_flip", False))
        brightness = float(config.get("brightness_jitter", 0))
        contrast = float(config.get("contrast_jitter", 0))

        result = image.copy()
        if rotation > 0:
            angle = random.uniform(-rotation, rotation)
            matrix = cv2.getRotationMatrix2D((14, 14), angle, 1.0)
            result = cv2.warpAffine(result, matrix, (28, 28), borderMode=cv2.BORDER_REPLICATE)
        if flip and random.random() < 0.5:
            result = cv2.flip(result, 1)
        if brightness or contrast:
            alpha = random.uniform(max(0.0, 1.0 - contrast), 1.0 + contrast)
            beta = random.uniform(-255.0 * brightness, 255.0 * brightness)
            result = cv2.convertScaleAbs(result, alpha=alpha, beta=beta)
        return result
