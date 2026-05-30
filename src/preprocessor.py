"""Image preprocessing utilities for training and live inference."""

from __future__ import annotations

import random
from pathlib import Path

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
    def prepare_word_video(
        video_path: str | Path,
        frame_count: int = 16,
        image_size: int = 64,
        frame_start: int = 1,
        frame_end: int = -1,
        bbox: list[int] | tuple[int, int, int, int] | None = None,
    ) -> torch.Tensor:
        """Prepare a sign video as a fixed-length word-classification tensor."""
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        try:
            frames: list[np.ndarray] = []
            max_decode_frames = max(int(frame_count), int(frame_count) * 3)
            start_index = max(0, int(frame_start) - 1)
            if start_index:
                capture.set(cv2.CAP_PROP_POS_FRAMES, start_index)
            while len(frames) < max_decode_frames:
                current_index = int(capture.get(cv2.CAP_PROP_POS_FRAMES))
                if int(frame_end) > 0 and current_index > int(frame_end):
                    break
                ok, frame = capture.read()
                if not ok or frame is None:
                    break
                if bbox is not None:
                    frame = Preprocessor._crop_bbox(frame, bbox)
                frames.append(frame)
        finally:
            capture.release()

        if not frames:
            raise ValueError(f"Video has no readable frames: {video_path}")
        if len(frames) > int(frame_count):
            indices = np.linspace(0, len(frames) - 1, int(frame_count), dtype=int)
            frames = [frames[int(index)] for index in indices]
        return Preprocessor.prepare_word_frames(frames, frame_count=frame_count, image_size=image_size)

    @staticmethod
    def _crop_bbox(frame: np.ndarray, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
        height, width = frame.shape[:2]
        x1, y1, x2, y2 = [int(value) for value in bbox]
        x1 = max(0, min(x1, width - 1))
        y1 = max(0, min(y1, height - 1))
        x2 = max(x1 + 1, min(x2, width))
        y2 = max(y1 + 1, min(y2, height))
        return frame[y1:y2, x1:x2]

    @staticmethod
    def prepare_word_frames(frames: list[np.ndarray], frame_count: int = 16, image_size: int = 64) -> torch.Tensor:
        """Prepare sampled BGR/RGB/grayscale frames for ``WordSignNet``."""
        if not frames:
            raise ValueError("frames cannot be empty")
        prepared: list[np.ndarray] = []
        source = frames[: int(frame_count)]
        while len(source) < int(frame_count):
            source.append(source[-1])
        for frame in source:
            if frame.ndim == 3:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            elif frame.ndim == 2:
                gray = frame
            else:
                raise ValueError(f"Unsupported frame shape: {frame.shape}")
            resized = cv2.resize(gray.astype(np.uint8), (int(image_size), int(image_size)), interpolation=cv2.INTER_AREA)
            prepared.append(resized.astype(np.float32) / 255.0)
        return torch.from_numpy(np.asarray(prepared, dtype=np.float32)).unsqueeze(1)

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
