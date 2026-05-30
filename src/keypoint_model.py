"""MediaPipe keypoint extraction and sequence model for word signs."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch import nn


POSE_LANDMARKS = 33
HAND_LANDMARKS = 21
KEYPOINT_DIM = POSE_LANDMARKS * 4 + HAND_LANDMARKS * 3 * 2


class MediaPipeKeypointExtractor:
    """Extract pose and hand keypoint sequences from frames or videos."""

    def __init__(self, frame_count: int = 32) -> None:
        self.frame_count = int(frame_count)
        cache_dir = Path("logs") / "matplotlib_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        os.environ.setdefault("MPLCONFIGDIR", str(cache_dir.resolve()))
        try:
            import mediapipe as mp
        except ImportError as exc:
            raise RuntimeError("mediapipe is required for keypoint word training/inference") from exc
        self.mp_holistic = mp.solutions.holistic
        self.holistic = self.mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=1,
            smooth_landmarks=True,
            enable_segmentation=False,
            refine_face_landmarks=False,
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
        )

    def close(self) -> None:
        self.holistic.close()

    def frame_keypoints(self, frame: np.ndarray) -> np.ndarray:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        return _results_to_vector(self.holistic.process(rgb))

    def video_keypoints(
        self,
        video_path: str | Path,
        frame_start: int = 1,
        frame_end: int = -1,
        bbox: list[int] | tuple[int, int, int, int] | None = None,
    ) -> np.ndarray:
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise ValueError(f"Could not open video: {video_path}")
        vectors: list[np.ndarray] = []
        try:
            start_index = max(0, int(frame_start) - 1)
            total_frames = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            if total_frames > 0:
                end_index = min(total_frames - 1, int(frame_end) - 1) if int(frame_end) > 0 else total_frames - 1
                if end_index < start_index:
                    end_index = total_frames - 1
                frame_indices = np.linspace(start_index, end_index, self.frame_count, dtype=int)
            else:
                frame_indices = np.arange(start_index, start_index + self.frame_count, dtype=int)

            for frame_index in frame_indices:
                capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
                ok, frame = capture.read()
                if not ok or frame is None:
                    continue
                if bbox is not None:
                    frame = _crop_bbox(frame, bbox)
                vectors.append(self.frame_keypoints(frame))
        finally:
            capture.release()
        if not vectors:
            raise ValueError(f"Video has no readable frames: {video_path}")
        return sample_keypoint_sequence(np.asarray(vectors, dtype=np.float32), self.frame_count)


def keypoint_cache_path(cache_dir: str | Path, video_path: Path, frame_start: int, frame_end: int, bbox: Any) -> Path:
    """Build a stable cache path for one video plus extraction settings."""
    payload = json.dumps(
        {"path": str(video_path.as_posix()), "frame_start": int(frame_start), "frame_end": int(frame_end), "bbox": bbox},
        sort_keys=True,
    )
    return Path(cache_dir) / f"{hashlib.sha1(payload.encode('utf-8')).hexdigest()}.npy"


def sample_keypoint_sequence(sequence: np.ndarray, frame_count: int) -> np.ndarray:
    """Sample or pad a keypoint sequence to a fixed length."""
    if sequence.ndim != 2 or sequence.shape[1] != KEYPOINT_DIM:
        raise ValueError(f"Expected keypoint sequence shape (T, {KEYPOINT_DIM}), got {tuple(sequence.shape)}")
    if len(sequence) >= int(frame_count):
        indices = np.linspace(0, len(sequence) - 1, int(frame_count), dtype=int)
        return sequence[indices].astype(np.float32)
    padding = np.repeat(sequence[-1:], int(frame_count) - len(sequence), axis=0)
    return np.concatenate((sequence, padding), axis=0).astype(np.float32)


class KeypointWordNet(nn.Module):
    """Temporal model over MediaPipe pose and hand keypoint sequences."""

    architecture = "mediapipe_keypoint_tcn_v1"

    def __init__(self, num_classes: int, dropout_rate: float = 0.4, input_dim: int = KEYPOINT_DIM) -> None:
        super().__init__()
        if num_classes <= 0:
            raise ValueError("KeypointWordNet requires at least one class")
        self.input_norm = nn.LayerNorm(input_dim)
        self.input_projection = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
        )
        self.temporal = nn.Sequential(
            _TemporalBlock(256, dilation=1, dropout_rate=dropout_rate),
            _TemporalBlock(256, dilation=2, dropout_rate=dropout_rate),
            _TemporalBlock(256, dilation=4, dropout_rate=dropout_rate),
            _TemporalBlock(256, dilation=8, dropout_rate=dropout_rate),
        )
        self.attention = nn.Sequential(nn.Conv1d(256, 128, kernel_size=1), nn.Tanh(), nn.Conv1d(128, 1, kernel_size=1))
        self.classifier = nn.Sequential(
            nn.Linear(512, 384),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Linear(384, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim != 3 or x.shape[2] != KEYPOINT_DIM:
            raise ValueError(f"Expected input shape (B, T, {KEYPOINT_DIM}), got {tuple(x.shape)}")
        temporal = self.temporal(self.input_projection(self.input_norm(x)).transpose(1, 2))
        weights = torch.softmax(self.attention(temporal), dim=2)
        attended = (temporal * weights).sum(dim=2)
        pooled = temporal.max(dim=2).values
        return self.classifier(torch.cat((attended, pooled), dim=1))

    def predict(self, x: torch.Tensor) -> tuple[int, float]:
        class_index, confidence, _ = self.predict_with_margin(x)
        return class_index, confidence

    def predict_with_margin(self, x: torch.Tensor) -> tuple[int, float, float]:
        self.eval()
        with torch.no_grad():
            probs = torch.softmax(self.forward(x), dim=1)
            top_values, top_indices = probs.topk(k=min(2, probs.shape[1]), dim=1)
        confidence = float(top_values[0, 0].item())
        runner_up = float(top_values[0, 1].item()) if top_values.shape[1] > 1 else 0.0
        return int(top_indices[0, 0].item()), confidence, confidence - runner_up


class _TemporalBlock(nn.Module):
    def __init__(self, channels: int, dilation: int, dropout_rate: float) -> None:
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_rate),
            nn.Conv1d(channels, channels, kernel_size=3, padding=dilation, dilation=dilation),
            nn.BatchNorm1d(channels),
        )
        self.activation = nn.ReLU(inplace=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(x + self.block(x))


def _results_to_vector(results: Any) -> np.ndarray:
    values: list[float] = []
    values.extend(_landmarks_to_values(getattr(results, "pose_landmarks", None), POSE_LANDMARKS, include_visibility=True))
    values.extend(_landmarks_to_values(getattr(results, "left_hand_landmarks", None), HAND_LANDMARKS, include_visibility=False))
    values.extend(_landmarks_to_values(getattr(results, "right_hand_landmarks", None), HAND_LANDMARKS, include_visibility=False))
    return np.asarray(values, dtype=np.float32)


def _landmarks_to_values(landmarks: Any, count: int, include_visibility: bool) -> list[float]:
    width = 4 if include_visibility else 3
    if landmarks is None:
        return [0.0] * count * width
    selected = landmarks.landmark[:count]
    values: list[float] = []
    for landmark in selected:
        values.extend([float(landmark.x), float(landmark.y), float(landmark.z)])
        if include_visibility:
            values.append(float(getattr(landmark, "visibility", 0.0)))
    values.extend([0.0] * (count - len(selected)) * width)
    return values


def _crop_bbox(frame: np.ndarray, bbox: list[int] | tuple[int, int, int, int]) -> np.ndarray:
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(value) for value in bbox]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    return frame[y1:y2, x1:x2]
