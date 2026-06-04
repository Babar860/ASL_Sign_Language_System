"""Live webcam recognition engine for Sign-to-Text mode."""

from __future__ import annotations

from typing import Any

import cv2
import numpy as np
import torch

from src.keypoint_model import KEYPOINT_DIM, KeypointWordNet, MediaPipeKeypointExtractor, keypoint_motion_features, normalize_keypoint_sequence
from src.model import SignNet, index_to_letter
from src.preprocessor import Preprocessor
from src.word_model import LegacyWordSignNet, WORD_MODEL_TYPES, WordSignNet

WORD_RECOGNIZER_TYPES = WORD_MODEL_TYPES + (KeypointWordNet,)


class Recognizer:
    """Capture webcam frames, classify the ROI, and manage sentence text."""

    def __init__(self, config: dict[str, Any], model: SignNet | WordSignNet | LegacyWordSignNet | KeypointWordNet) -> None:
        """Create a recognizer.

        Args:
            config: Validated application config.
            model: Loaded SignNet model.
        """
        self.config = config
        self.model = model
        self.device = next(model.parameters()).device
        self.cap: cv2.VideoCapture | None = None
        self.sentence_buffer = ""
        self.current_prediction = ""
        self.word_frame_buffer: list[Any] = []
        self.keypoint_buffer: list[np.ndarray] = []
        self.keypoint_extractor: MediaPipeKeypointExtractor | None = None
        self.recent_word_predictions: list[str] = []
        self.active = False
        print(f"Using device: {self.device}")

    def start(self) -> None:
        """Open the configured webcam and initialize the session."""
        self.sentence_buffer = ""
        self.current_prediction = ""
        self.word_frame_buffer = []
        self.keypoint_buffer = []
        self.recent_word_predictions = []
        if isinstance(self.model, KeypointWordNet):
            self.keypoint_extractor = MediaPipeKeypointExtractor(frame_count=int(self.config["model"]["word_frame_count"]))
        self.cap = cv2.VideoCapture(int(self.config["inference"]["camera_index"]))
        self.cap.set(cv2.CAP_PROP_FPS, int(self.config["inference"]["target_fps"]))
        if not self.cap.isOpened():
            raise RuntimeError("Camera not available")
        self.active = True

    def stop(self) -> None:
        """Stop recognition and release the webcam when one is open."""
        self.active = False
        cap = self.cap
        self.cap = None
        if cap is not None:
            cap.release()
        if self.keypoint_extractor is not None:
            self.keypoint_extractor.close()
            self.keypoint_extractor = None

    def step(self) -> Any:
        """Process one camera frame and return an annotated BGR frame."""
        if self.cap is None or not self.cap.isOpened():
            raise RuntimeError("Camera not available")
        ok, frame = self.cap.read()
        if not ok or frame is None:
            raise RuntimeError("Camera frame not available")

        if isinstance(self.model, WORD_RECOGNIZER_TYPES):
            display = self._predict_word_frame(frame)
            self.current_prediction = display
            annotated = frame.copy()
            cv2.putText(annotated, display, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        else:
            roi, coords = self.extract_roi(frame)
            display = self._predict_roi(roi)
            self.current_prediction = display
            annotated = self.draw_roi_rectangle(frame.copy(), coords)
            cv2.putText(annotated, display, (coords[0], max(30, coords[1] - 12)), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.putText(annotated, self.sentence_buffer, (20, annotated.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        return annotated

    def _predict_word_frame(self, frame: Any) -> str:
        if isinstance(self.model, KeypointWordNet):
            return self._predict_keypoint_frame(frame)

        self.word_frame_buffer.append(frame.copy())
        frame_count = int(self.config["model"]["word_frame_count"])
        self.word_frame_buffer = self.word_frame_buffer[-frame_count:]
        if len(self.word_frame_buffer) < frame_count:
            return "Collecting..."
        tensor = Preprocessor.prepare_word_frames(
            self.word_frame_buffer,
            frame_count=frame_count,
            image_size=int(self.config["model"]["word_image_size"]),
        ).unsqueeze(0).to(self.device)
        return self._word_label_from_tensor(tensor)

    def _predict_roi(self, roi: Any) -> str:
        if isinstance(self.model, WORD_RECOGNIZER_TYPES):
            return self._predict_word_frame(roi)

        tensor = Preprocessor.prepare_frame(roi).to(self.device)
        class_index, confidence = self.model.predict(tensor)
        if confidence >= float(self.config["inference"]["confidence_threshold"]):
            return index_to_letter(class_index)
        return "Low Confidence"

    def _predict_keypoint_frame(self, frame: Any) -> str:
        if self.keypoint_extractor is None:
            self.keypoint_extractor = MediaPipeKeypointExtractor(frame_count=int(self.config["model"]["word_frame_count"]))
        self.keypoint_buffer.append(self.keypoint_extractor.frame_keypoints(frame))
        frame_count = int(self.config["model"]["word_frame_count"])
        self.keypoint_buffer = self.keypoint_buffer[-frame_count:]
        if len(self.keypoint_buffer) < frame_count:
            return "Collecting..."
        sequence = keypoint_motion_features(np.asarray(self.keypoint_buffer, dtype=np.float32))
        tensor = torch.from_numpy(sequence).unsqueeze(0).to(self.device)
        return self._word_label_from_tensor(tensor)

    def _word_label_from_tensor(self, tensor: torch.Tensor) -> str:
        class_index, confidence, margin = self.model.predict_with_margin(tensor)
        labels = getattr(self.model, "word_labels", [])
        label = str(labels[class_index]) if labels else str(class_index)
        self.recent_word_predictions.append(label)
        stability_frames = int(self.config["model"]["word_stability_frames"])
        self.recent_word_predictions = self.recent_word_predictions[-stability_frames:]
        stable = len(self.recent_word_predictions) == stability_frames and len(set(self.recent_word_predictions)) == 1
        if (
            stable
            and confidence >= float(self.config["model"]["word_confidence_threshold"])
            and margin >= float(self.config["model"]["word_margin_threshold"])
        ):
            return label
        return "Low Confidence"

    def extract_roi(self, frame: Any) -> tuple[Any, tuple[int, int, int, int]]:
        """Extract the configured ROI from the center or configured coordinates."""
        height, width = frame.shape[:2]
        size = min(int(self.config["inference"]["roi_size"]), width, height)
        roi_x = self.config["inference"].get("roi_x")
        roi_y = self.config["inference"].get("roi_y")
        x1 = int(roi_x) if roi_x is not None else (width - size) // 2
        y1 = int(roi_y) if roi_y is not None else (height - size) // 2
        x1 = max(0, min(x1, width - size))
        y1 = max(0, min(y1, height - size))
        x2 = x1 + size
        y2 = y1 + size
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)

    def draw_roi_rectangle(self, frame: Any, coords: tuple[int, int, int, int] | None = None) -> Any:
        """Draw the ROI rectangle on a frame."""
        if coords is None:
            _, coords = self.extract_roi(frame)
        x1, y1, x2, y2 = coords
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame

    def append_letter(self, letter: str) -> None:
        """Append a recognized letter to the sentence buffer."""
        self.sentence_buffer += letter[:1]

    def show_current_prediction(self) -> bool:
        """Append the current prediction when the user accepts the sign.

        Returns:
            ``True`` when a single predicted letter was appended, otherwise
            ``False`` for low-confidence or empty predictions.
        """
        if isinstance(self.model, WORD_RECOGNIZER_TYPES):
            if self.current_prediction not in {"", "Collecting...", "Low Confidence"}:
                if self.sentence_buffer and not self.sentence_buffer.endswith(" "):
                    self.sentence_buffer += " "
                self.sentence_buffer += self.current_prediction + " "
                return True
            return False

        if len(self.current_prediction) == 1 and self.current_prediction.isalpha():
            self.append_letter(self.current_prediction)
            return True
        return False

    def append_space(self) -> None:
        """Append a space to the sentence buffer."""
        self.sentence_buffer += " "

    def finalize_word(self) -> None:
        """Finalize the current word by ensuring a trailing space."""
        if self.sentence_buffer and not self.sentence_buffer.endswith(" "):
            self.sentence_buffer += " "

    def backspace(self) -> None:
        """Remove the last character from the sentence buffer."""
        self.sentence_buffer = self.sentence_buffer[:-1]

    def clear(self) -> None:
        """Clear the sentence buffer."""
        self.sentence_buffer = ""


def load_model_for_inference(config: dict[str, Any]) -> SignNet | WordSignNet | LegacyWordSignNet | KeypointWordNet:
    """Load word-level weights when present, otherwise static-letter weights."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    word_checkpoint_path = config["model"].get("word_checkpoint_path")
    if word_checkpoint_path and torch.jit.is_scripting() is False:
        from pathlib import Path

        path = Path(word_checkpoint_path)
        if path.exists():
            checkpoint = torch.load(path, map_location=device)
            labels = checkpoint.get("labels", []) if isinstance(checkpoint, dict) else []
            state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
            architecture = checkpoint.get("architecture", "") if isinstance(checkpoint, dict) else ""
            if str(architecture).startswith("mediapipe_keypoint"):
                model = KeypointWordNet(
                    num_classes=len(labels),
                    dropout_rate=float(config["model"]["dropout_rate"]),
                    input_dim=int(checkpoint.get("input_dim", KEYPOINT_DIM)) if isinstance(checkpoint, dict) else KEYPOINT_DIM,
                ).to(device)
                model.load_state_dict(state)
            else:
                model = WordSignNet(
                    num_classes=len(labels),
                    dropout_rate=float(config["model"]["dropout_rate"]),
                ).to(device)
                try:
                    model.load_state_dict(state)
                except RuntimeError:
                    model = LegacyWordSignNet(
                        num_classes=len(labels),
                        dropout_rate=float(config["model"]["dropout_rate"]),
                    ).to(device)
                    model.load_state_dict(state)
            model.word_labels = labels
            model.eval()
            return model

    model = SignNet(
        num_classes=int(config["model"]["num_classes"]),
        dropout_rate=float(config["model"]["dropout_rate"]),
    ).to(device)
    checkpoint = torch.load(config["model"]["checkpoint_path"], map_location=device)
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    model.load_state_dict(state)
    model.eval()
    return model
